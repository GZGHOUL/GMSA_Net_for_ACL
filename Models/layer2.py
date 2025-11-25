"""
增强版 Layer2 网络
核心改进：双流对称架构 + 物理约束 + 类别平衡
"""

import torch
import torch.nn as nn
from Models.clinical_modules import (
    GaitSymmetryIndex,
    KneeInstabilityDetector,
    PhysicsGuidedAttention,
    CrossAttention,
    ROMAnalyzer,
    LoadingPatternExtractor
)
from Models.Global_Multi_scale_Attention import GMSA_Block
from Models.Attention import Attention, CrossAttention
from Models.shapeformer import ShapeBlock, LearnablePositionEncoding



class Layer2_DualStream(nn.Module):
    """
    双流对称网络（方案A）
    
    设计理念：
    1. 左右腿独立流提取特征
    2. 交叉注意力建模对称性
    3. 差异编码捕获异常
    4. 类别平衡机制
    
    适用于：前交叉韧带损伤的半月板损伤二分类
    """
    def __init__(self, config):
        super().__init__()
        
        # 多尺度卷积核配置
        # 涵盖不同时空尺度：单通道、3通道、6通道（半腿） × 不同时间窗口
        self.ks = [
            [1,25], [1,50], [1,100],   # 单通道，不同时间窗
            [3,25], [3,50], [3,100],   # 3通道组合
            [6,25], [6,50], [6,100]    # 整条腿
        ]
        
        # ========== 阶段1：左右腿独立流 ==========
        self.left_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        # 右腿流（权重独立，捕获患侧/健侧差异）
        self.right_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        d = self.left_blocks[0].out_dim  # 每个block输出维度（通常256）
        num_blocks = len(self.ks)
        
        # ========== 阶段2：Gating机制（每条腿独立） ==========
        self.left_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Softmax(dim=-1)
        )
        
        self.right_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Softmax(dim=-1)
        )
        
        # ========== 阶段3：对称性与差异性建模 ==========
        # 交叉注意力：左→右，右→左
        self.cross_attn_lr = CrossAttention(dim=d, num_heads=4, dropout=config['dropout'])
        self.cross_attn_rl = CrossAttention(dim=d, num_heads=4, dropout=config['dropout'])
        
        # 差异编码器
        self.diff_encoder = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.LayerNorm(d // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d // 2, d // 2)
        )
        
        # ========== 阶段4：类别平衡分类头 ==========
        total_dim = d * 2 + d // 2  # left + right + diff
        
        # 每个类别的专家网络
        self.class_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256)
            ) for _ in range(2)  # 2个类别
        ])
        
        # 最终分类器
        self.final_classifier = nn.Linear(256 * 2, 2)
        
        # 对比学习投影头（可选，用于训练时的对比损失）
        self.projection_head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
    
    def _adapt_config_for_single_leg(self, config):
        """调整config使其适用于单腿（6个自由度）"""
        adapted_config = config.copy()
        adapted_config['ts_dim'] = 6  # 单腿6个自由度
        return adapted_config
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 12, 600] 完整步态数据
            return_features: 是否返回中间特征（用于对比学习）
        
        Returns:
            logits: [B, 2] 分类logits
            features (optional): 中间特征字典
        """
        B = x.shape[0]
        
        # ========== 分离左右腿 ==========
        left_x = x[:, :6, :]   # [B, 6, 600]
        right_x = x[:, 6:, :]  # [B, 6, 600]
        
        # ========== 阶段1：双流特征提取 ==========
        # 左腿
        left_feats = [blk(left_x) for blk in self.left_blocks]  # num_blocks × [B, d]
        left_cat = torch.cat(left_feats, dim=-1)  # [B, d*num_blocks]
        
        # 右腿
        right_feats = [blk(right_x) for blk in self.right_blocks]  # num_blocks × [B, d]
        right_cat = torch.cat(right_feats, dim=-1)  # [B, d*num_blocks]
        
        # ========== 阶段2：Gating加权融合 ==========
        # 左腿gating
        left_gate_weights = self.left_gate(left_cat)  # [B, num_blocks]
        left_stack = torch.stack(left_feats, dim=1)   # [B, num_blocks, d]
        left_fused = (left_stack * left_gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, d]
        
        # 右腿gating
        right_gate_weights = self.right_gate(right_cat)  # [B, num_blocks]
        right_stack = torch.stack(right_feats, dim=1)   # [B, num_blocks, d]
        right_fused = (right_stack * right_gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, d]
        
        # ========== 阶段3：对称性与差异性 ==========
        # 交叉注意力增强（建模左右腿的相互影响）
        left_enhanced = left_fused + self.cross_attn_lr(
            left_fused.unsqueeze(1), 
            right_fused.unsqueeze(1)
        ).squeeze(1)
        
        right_enhanced = right_fused + self.cross_attn_rl(
            right_fused.unsqueeze(1), 
            left_fused.unsqueeze(1)
        ).squeeze(1)
        
        # 差异编码（捕获左右腿的异常差异）
        diff = torch.abs(left_enhanced - right_enhanced)
        diff_encoded = self.diff_encoder(diff)  # [B, d//2]
        
        # ========== 阶段4：融合与分类 ==========
        # 总特征
        total_feat = torch.cat([left_enhanced, right_enhanced, diff_encoded], dim=-1)
        
        # 类别专家
        expert_outputs = [expert(total_feat) for expert in self.class_experts]
        expert_concat = torch.cat(expert_outputs, dim=-1)  # [B, 256*2]
        
        # 最终分类
        logits = self.final_classifier(expert_concat)
        
        if return_features:
            projection = self.projection_head(total_feat)
            features = {
                'left_fused': left_fused,
                'right_fused': right_fused,
                'left_enhanced': left_enhanced,
                'right_enhanced': right_enhanced,
                'diff_encoded': diff_encoded,
                'projection': projection,
                'left_gate_weights': left_gate_weights,
                'right_gate_weights': right_gate_weights
            }
            return logits, features
        
        return logits


class Layer2_PhysicsGuided(nn.Module):
    """
    物理约束引导网络（方案B）
    
    在双流基础上增加物理约束注意力
    """
    def __init__(self, config):
        super().__init__()
        
        # ========== 物理约束预处理 ==========
        self.physics_attn_left = PhysicsGuidedAttention(num_dof=6)
        self.physics_attn_right = PhysicsGuidedAttention(num_dof=6)
        
        # ========== 多尺度卷积核 ==========
        self.ks = [[1,25], [1,50], [1,100], [3,25], [3,50], [3,100], [6,25], [6,50], [6,100]]
        
        # ========== 双流网络 ==========
        self.left_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        self.right_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        d = self.left_blocks[0].out_dim
        num_blocks = len(self.ks)
        
        # ========== Gating ==========
        self.left_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks)
        )
        
        self.right_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks)
        )
        
        # ========== 对称性与差异性 ==========
        self.cross_attn_lr = CrossAttention(dim=d, num_heads=4, dropout=config['dropout'])
        self.cross_attn_rl = CrossAttention(dim=d, num_heads=4, dropout=config['dropout'])
        
        self.diff_encoder = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.LayerNorm(d // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d // 2, d // 2)
        )
        
        # ========== 分类头 ==========
        total_dim = d * 2 + d // 2
        
        self.class_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256)
            ) for _ in range(2)
        ])
        
        self.final_classifier = nn.Linear(256 * 2, 2)
    
    def _adapt_config_for_single_leg(self, config):
        adapted_config = config.copy()
        adapted_config['ts_dim'] = 6
        return adapted_config
    
    def forward(self, x, return_interpretation=False):
        """
        Args:
            x: [B, 12, 600]
            return_interpretation: 是否返回可解释性信息
        
        Returns:
            logits: [B, 2]
            interpretation (optional): 包含物理耦合权重等信息
        """
        B = x.shape[0]
        
        # 分离左右腿
        left_x = x[:, :6, :]
        right_x = x[:, 6:, :]
        
        # ========== 物理约束注意力 ==========
        left_x_weighted, left_coupling = self.physics_attn_left(left_x)
        right_x_weighted, right_coupling = self.physics_attn_right(right_x)
        
        # ========== 双流特征提取 ==========
        left_feats = [blk(left_x_weighted) for blk in self.left_blocks]
        left_cat = torch.cat(left_feats, dim=-1)
        
        right_feats = [blk(right_x_weighted) for blk in self.right_blocks]
        right_cat = torch.cat(right_feats, dim=-1)
        
        # ========== Gating ==========
        left_gate_weights = torch.softmax(self.left_gate(left_cat), dim=-1)
        left_stack = torch.stack(left_feats, dim=1)
        left_fused = (left_stack * left_gate_weights.unsqueeze(-1)).sum(dim=1)
        
        right_gate_weights = torch.softmax(self.right_gate(right_cat), dim=-1)
        right_stack = torch.stack(right_feats, dim=1)
        right_fused = (right_stack * right_gate_weights.unsqueeze(-1)).sum(dim=1)
        
        # ========== 对称性与差异性 ==========
        left_enhanced = left_fused + self.cross_attn_lr(
            left_fused.unsqueeze(1), 
            right_fused.unsqueeze(1)
        ).squeeze(1)
        
        right_enhanced = right_fused + self.cross_attn_rl(
            right_fused.unsqueeze(1), 
            left_fused.unsqueeze(1)
        ).squeeze(1)
        
        diff = torch.abs(left_enhanced - right_enhanced)
        diff_encoded = self.diff_encoder(diff)
        
        # ========== 分类 ==========
        total_feat = torch.cat([left_enhanced, right_enhanced, diff_encoded], dim=-1)
        
        expert_outputs = [expert(total_feat) for expert in self.class_experts]
        expert_concat = torch.cat(expert_outputs, dim=-1)
        
        logits = self.final_classifier(expert_concat)
        
        if return_interpretation:
            interpretation = {
                'left_coupling': left_coupling,    # [B, 6] 左腿物理耦合权重
                'right_coupling': right_coupling,  # [B, 6] 右腿物理耦合权重
                'left_gate': left_gate_weights,    # [B, 9] 左腿多尺度门控
                'right_gate': right_gate_weights,  # [B, 9] 右腿多尺度门控
                'asymmetry': diff_encoded          # [B, d//2] 左右腿差异
            }
            return logits, interpretation
        
        return logits


class Layer2_ClinicalEnhanced(nn.Module):
    """
    临床特征增强网络（综合方案）
    
    结合：
    1. 双流对称架构
    2. 物理约束
    3. 临床特征（对称性、不稳定性、ROM、负重模式）
    4. 类别平衡
    """
    def __init__(self, config):
        super().__init__()
        
        # ========== 物理约束 ==========
        self.physics_attn_left = PhysicsGuidedAttention(num_dof=6)
        self.physics_attn_right = PhysicsGuidedAttention(num_dof=6)
        
        # ========== 双流多尺度网络 ==========
        self.ks = [[1,25], [1,50], [1,100], [3,25], [3,50], [3,100]]  # 减少为6个
        
        self.left_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        self.right_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        d = self.left_blocks[0].out_dim
        num_blocks = len(self.ks)
        
        # ========== Gating ==========
        self.left_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks)
        )
        
        self.right_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks)
        )
        
        # ========== 对称性与差异性 ==========
        self.cross_attn_lr = CrossAttention(dim=d, num_heads=4, dropout=config['dropout'])
        self.cross_attn_rl = CrossAttention(dim=d, num_heads=4, dropout=config['dropout'])
        
        self.diff_encoder = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.LayerNorm(d // 2),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )
        
        # ========== 临床特征提取器 ==========
        self.gait_symmetry = GaitSymmetryIndex(output_dim=32)
        self.instability_detector = KneeInstabilityDetector(output_dim=32)
        self.rom_analyzer = ROMAnalyzer(output_dim=32)
        self.loading_extractor = LoadingPatternExtractor(output_dim=32)
        
        # ========== 特征融合 ==========
        clinical_dim = 32 * 4  # 4个临床特征
        total_dim = d * 2 + d // 2 + clinical_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # ========== 类别平衡分类头 ==========
        self.class_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128)
            ) for _ in range(2)
        ])
        
        self.final_classifier = nn.Linear(128 * 2, 2)
        
        # 对比学习投影
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
    
    def _adapt_config_for_single_leg(self, config):
        adapted_config = config.copy()
        adapted_config['ts_dim'] = 6
        return adapted_config
    
    def forward(self, x, return_all=False):
        """
        Args:
            x: [B, 12, 600]
            return_all: 返回所有特征（用于可解释性和对比学习）
        
        Returns:
            logits: [B, 2]
            outputs (optional): 包含所有中间特征和可解释性信息
        """
        B = x.shape[0]
        
        left_x = x[:, :6, :]
        right_x = x[:, 6:, :]
        
        # ========== 物理约束 ==========
        left_x_weighted, left_coupling = self.physics_attn_left(left_x)
        right_x_weighted, right_coupling = self.physics_attn_right(right_x)
        
        # ========== 双流特征 ==========
        left_feats = [blk(left_x_weighted) for blk in self.left_blocks]
        left_cat = torch.cat(left_feats, dim=-1)
        
        right_feats = [blk(right_x_weighted) for blk in self.right_blocks]
        right_cat = torch.cat(right_feats, dim=-1)
        
        # ========== Gating ==========
        left_gate_weights = torch.softmax(self.left_gate(left_cat), dim=-1)
        left_stack = torch.stack(left_feats, dim=1)
        left_fused = (left_stack * left_gate_weights.unsqueeze(-1)).sum(dim=1)
        
        right_gate_weights = torch.softmax(self.right_gate(right_cat), dim=-1)
        right_stack = torch.stack(right_feats, dim=1)
        right_fused = (right_stack * right_gate_weights.unsqueeze(-1)).sum(dim=1)
        
        # ========== 对称性与差异性 ==========
        left_enhanced = left_fused + self.cross_attn_lr(
            left_fused.unsqueeze(1), 
            right_fused.unsqueeze(1)
        ).squeeze(1)
        
        right_enhanced = right_fused + self.cross_attn_rl(
            right_fused.unsqueeze(1), 
            left_fused.unsqueeze(1)
        ).squeeze(1)
        
        diff = torch.abs(left_enhanced - right_enhanced)
        diff_encoded = self.diff_encoder(diff)
        
        # ========== 临床特征 ==========
        symmetry_feat = self.gait_symmetry(left_x, right_x)
        instability_feat = self.instability_detector(x)
        rom_feat = self.rom_analyzer(left_x)  # 可以用左腿或平均
        loading_feat = self.loading_extractor(x)
        
        clinical_feat = torch.cat([
            symmetry_feat, 
            instability_feat, 
            rom_feat, 
            loading_feat
        ], dim=-1)
        
        # ========== 融合 ==========
        all_feat = torch.cat([
            left_enhanced, 
            right_enhanced, 
            diff_encoded, 
            clinical_feat
        ], dim=-1)
        
        fused_feat = self.fusion(all_feat)
        
        # ========== 分类 ==========
        expert_outputs = [expert(fused_feat) for expert in self.class_experts]
        expert_concat = torch.cat(expert_outputs, dim=-1)
        
        logits = self.final_classifier(expert_concat)
        
        if return_all:
            projection = self.projection_head(fused_feat)
            outputs = {
                'logits': logits,
                'projection': projection,
                'left_coupling': left_coupling,
                'right_coupling': right_coupling,
                'symmetry_feat': symmetry_feat,
                'instability_feat': instability_feat,
                'rom_feat': rom_feat,
                'loading_feat': loading_feat,
                'diff_encoded': diff_encoded,
                'left_gate': left_gate_weights,
                'right_gate': right_gate_weights
            }
            return outputs
        
        return logits

class Layer2_AsyncDualStream_GMSA(nn.Module):
    """
    异步双流网络 + GMSA（适用于左右腿不同步数据）
    
    核心改进：
    1. 使用GMSA模块提取多尺度时空特征
    2. GMSA的AttnPool1D天然支持不同长度输入
    3. 在全局特征层面比较左右腿（而非逐时刻）
    4. 保留原有的多尺度+Gating架构优势
    """
    def __init__(self, config):
        super().__init__()
        
        # 多尺度卷积核配置（同原版）
        self.ks = [
            [1,25], [1,50], [1,100],   # 单通道，不同时间窗
            [3,25], [3,50], [3,100],   # 3通道组合
            [6,25], [6,50], [6,100]    # 整条腿
        ]
        
        # ========== 左右腿独立的GMSA流 ==========
        # 左腿GMSA blocks（独立）
        self.left_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        # 右腿GMSA blocks（独立，权重不共享）
        self.right_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        d = self.left_blocks[0].out_dim  # 每个block输出维度（通常256）
        num_blocks = len(self.ks)
        
        # ========== 独立的Gating机制 ==========
        # 左腿gating：从9个尺度中学习权重
        self.left_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Softmax(dim=-1)
        )
        
        # 右腿gating
        self.right_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Softmax(dim=-1)
        )
        
        # ========== 全局特征比较（异步安全）==========
        # 注意：这里不使用交叉注意力（因为左右腿不同步）
        # 而是在全局特征层面做比较
        
        # 左右腿特征的独立建模
        self.left_projector = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(0.6)
        )
        
        self.right_projector = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(0.6)
        )
        
        # 全局差异编码（基于统计量而非时序对齐）
        self.global_diff_encoder = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.LayerNorm(d // 2),
            nn.GELU(),
            nn.Dropout(0.6)
        )
        
        # ========== 统计特征增强（时间无关）==========
        # 提取每条腿的统计特征：均值、方差、最大、最小
        self.stat_feature_dim = 6 * 4  # 6个自由度 × 4个统计量
        self.stat_encoder = nn.Sequential(
            nn.Linear(self.stat_feature_dim * 2, 128),  # 左右腿统计量
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, 64)
        )
        
        # ========== 类别平衡分类头 ==========
        total_dim = d * 2 + d // 2 + 64  # left + right + diff + stat
        
        # 每个类别的专家网络
        self.class_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.6),
                nn.Linear(512, 256)
            ) for _ in range(2)
        ])
        
        # 最终分类器
        self.final_classifier = nn.Linear(256 * 2, 2)
        
        # 对比学习投影头（可选）
        self.projection_head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
    
    def _adapt_config_for_single_leg(self, config):
        """调整config使其适用于单腿（6个自由度）"""
        adapted_config = config.copy()
        adapted_config['ts_dim'] = 6  # 单腿6个自由度
        return adapted_config
    
    def _extract_statistical_features(self, x):
        """
        提取时间无关的统计特征
        x: [B, C, T]
        return: [B, C*4]
        """
        mean = x.mean(dim=2)  # [B, C]
        std = x.std(dim=2)    # [B, C]
        max_val = x.max(dim=2)[0]  # [B, C]
        min_val = x.min(dim=2)[0]  # [B, C]
        
        stats = torch.cat([mean, std, max_val, min_val], dim=-1)  # [B, C*4]
        return stats
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 12, T] 完整步态数据（左右腿可能不同步，但拼接在一起）
            return_features: 是否返回中间特征
        
        Returns:
            logits: [B, 2]
            features (optional): 中间特征字典
        """
        B = x.shape[0]
        
        # ========== 分离左右腿 ==========
        left_x = x[:, :6, :]   # [B, 6, T]
        right_x = x[:, 6:, :]  # [B, 6, T]
        
        # ========== 阶段1：GMSA多尺度特征提取 ==========
        # 左腿：9个不同尺度的GMSA block
        left_feats = []
        for blk in self.left_blocks:
            try:
                feat = blk(left_x)  # [B, d]
                left_feats.append(feat)
            except Exception as e:
                # 如果某个尺度的kernel_size大于时间步长，跳过
                print(f"Warning: Left leg GMSA block failed: {e}")
                continue
        
        # 右腿：9个不同尺度的GMSA block
        right_feats = []
        for blk in self.right_blocks:
            try:
                feat = blk(right_x)  # [B, d]
                right_feats.append(feat)
            except Exception as e:
                print(f"Warning: Right leg GMSA block failed: {e}")
                continue
        
        # 如果所有block都失败了，使用fallback
        if len(left_feats) == 0 or len(right_feats) == 0:
            raise ValueError("All GMSA blocks failed. Check input dimensions.")
        
        # ========== 阶段2：Gating加权融合 ==========
        # 左腿gating
        left_cat = torch.cat(left_feats, dim=-1)  # [B, d*num_valid_blocks]
        left_gate_weights = self.left_gate(left_cat)  # [B, num_blocks]
        left_stack = torch.stack(left_feats, dim=1)   # [B, num_blocks, d]
        left_fused = (left_stack * left_gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, d]
        
        # 右腿gating
        right_cat = torch.cat(right_feats, dim=-1)
        right_gate_weights = self.right_gate(right_cat)
        right_stack = torch.stack(right_feats, dim=1)
        right_fused = (right_stack * right_gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, d]
        
        # ========== 阶段3：全局特征投影 ==========
        # 注意：不使用交叉注意力，因为左右腿不同步
        left_projected = self.left_projector(left_fused)   # [B, d]
        right_projected = self.right_projector(right_fused) # [B, d]
        
        # ========== 阶段4：全局差异编码 ==========
        # 在全局特征层面计算差异（而非时间步层面）
        global_diff = torch.abs(left_projected - right_projected)
        diff_encoded = self.global_diff_encoder(global_diff)  # [B, d//2]
        
        # ========== 阶段5：统计特征 ==========
        # 提取时间无关的统计特征
        left_stats = self._extract_statistical_features(left_x)   # [B, 24]
        right_stats = self._extract_statistical_features(right_x) # [B, 24]
        all_stats = torch.cat([left_stats, right_stats], dim=-1)  # [B, 48]
        stat_features = self.stat_encoder(all_stats)  # [B, 64]
        
        # ========== 阶段6：融合与分类 ==========
        # 融合所有特征
        total_feat = torch.cat([
            left_projected,   # 左腿全局特征
            right_projected,  # 右腿全局特征
            diff_encoded,     # 左右腿差异
            stat_features     # 统计特征
        ], dim=-1)  # [B, d*2 + d//2 + 64]
        
        # 类别专家网络
        expert_outputs = [expert(total_feat) for expert in self.class_experts]
        expert_concat = torch.cat(expert_outputs, dim=-1)  # [B, 256*2]
        
        # 最终分类
        logits = self.final_classifier(expert_concat)  # [B, 2]
        
        if return_features:
            projection = self.projection_head(total_feat)
            features = {
                'left_fused': left_fused,
                'right_fused': right_fused,
                'left_projected': left_projected,
                'right_projected': right_projected,
                'diff_encoded': diff_encoded,
                'stat_features': stat_features,
                'projection': projection,
                'left_gate_weights': left_gate_weights,
                'right_gate_weights': right_gate_weights,
                'left_stats': left_stats,
                'right_stats': right_stats
            }
            return logits, features
        
        return logits

class Layer2_AsyncDualStream_Lite(nn.Module):
    """
    轻量版异步双流网络
    
    针对小数据集（<200样本）优化：
    - 减少GMSA blocks: 9个 → 3个
    - 减小隐藏层维度
    - 增加Dropout
    - 移除专家网络
    """
    def __init__(self, config):
        super().__init__()
        
        # ========== 只保留3个关键尺度 ==========
        self.ks = [
            [1,50],   # 单通道，中等时间窗
            [3,50],   # 3通道组合
            [6,50]    # 整条腿
        ]
        
        # 左右腿GMSA（减少到3个）
        self.left_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        self.right_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        d = self.left_blocks[0].out_dim
        num_blocks = len(self.ks)
        
        # ========== 简化的Gating ==========
        self.left_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Dropout(0.3),  # Gating也加dropout
            nn.Softmax(dim=-1)
        )
        
        self.right_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Dropout(0.3),
            nn.Softmax(dim=-1)
        )
        
        # ========== 简化的特征投影 ==========
        self.feature_fusion = nn.Sequential(
            nn.Linear(d * 2, 256),  # 左+右 → 256
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.6),  # 高dropout
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.6)
        )
        
        # ========== 简单分类头（不用专家网络）==========
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.7),  # 非常高的dropout
            nn.Linear(64, 2)
        )
    
    def _adapt_config_for_single_leg(self, config):
        adapted_config = config.copy()
        adapted_config['ts_dim'] = 6
        return adapted_config
    
    def forward(self, x, return_features=False):
        B = x.shape[0]
        
        left_x = x[:, :6, :]
        right_x = x[:, 6:, :]
        
        # GMSA特征提取
        left_feats = [blk(left_x) for blk in self.left_blocks]
        right_feats = [blk(right_x) for blk in self.right_blocks]
        
        # Gating融合
        left_cat = torch.cat(left_feats, dim=-1)
        left_gate_weights = self.left_gate(left_cat)
        left_stack = torch.stack(left_feats, dim=1)
        left_fused = (left_stack * left_gate_weights.unsqueeze(-1)).sum(dim=1)
        
        right_cat = torch.cat(right_feats, dim=-1)
        right_gate_weights = self.right_gate(right_cat)
        right_stack = torch.stack(right_feats, dim=1)
        right_fused = (right_stack * right_gate_weights.unsqueeze(-1)).sum(dim=1)
        
        # 融合左右腿
        combined = torch.cat([left_fused, right_fused], dim=-1)
        features = self.feature_fusion(combined)
        
        # 分类
        logits = self.classifier(features)
        
        if return_features:
            return logits, {
                'left_fused': left_fused,
                'right_fused': right_fused,
                'features': features
            }
        
        return logits

class Layer2_UltraLite(nn.Module):
    """
    超轻量网络 - 专为<200样本设计
    
    核心思想：
    1. 不用GMSA（太重了）
    2. 只用简单的1D CNN + 全局池化
    3. 极简分类头
    """
    def __init__(self, config):
        super().__init__()
        
        # ========== 轻量特征提取器 ==========
        # 左腿提取器
        self.left_encoder = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3),
            
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3),
            
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.AdaptiveAvgPool1d(1)  # 全局池化 → [B, 64, 1]
        )
        
        # 右腿提取器（权重共享以减少参数）
        self.right_encoder = self.left_encoder  # 共享权重
        
        # ========== 超简单分类头 ==========
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2, 64),  # 左+右=128 → 64
            nn.Dropout(0.7),
            nn.GELU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x, return_features=False):
        left_x = x[:, :6, :]   # [B, 6, T]
        right_x = x[:, 6:, :]  # [B, 6, T]
        
        # 提取特征
        left_feat = self.left_encoder(left_x)    # [B, 64, 1]
        right_feat = self.right_encoder(right_x) # [B, 64, 1]
        
        # 拼接
        combined = torch.cat([left_feat, right_feat], dim=1)  # [B, 128, 1]
        
        # 分类
        logits = self.classifier(combined)
        
        return logits


class Layer2_shapeformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 存储形状子信息（【样本索引， 起始帧， 终止帧， 信息增益权重， 类别， 维度】）和形状子序列
        self.shapelets_info = config['shapelets_info_layer2']
        self.shapelets_info = torch.IntTensor(self.shapelets_info)
        self.shapelets = config['shapelets_layer2']

        # 形状子权重参数（可学习）：基于IG初始化，训练中动态调整形状子重要性
        self.shape_weight = torch.nn.Parameter(torch.tensor(config['shapelets_info_layer2'][:, 3]).float(),
                                               requires_grad=True)

        # Class-specific Transformer
        self.shapeblock = nn.ModuleList([
            ShapeBlock(shapelet_info=self.shapelets_info[i], shapelet=self.shapelets[i],
                       shape_embed_dim=config['shape_embed_dim'],
                       len_window_shapeblock=config['len_window_shapeblock'], len_ts=config['len_ts'],
                       norm=config['norm'], max_ci=config['max_ci'])
            for i in range(len(self.shapelets_info))
        ])

        self.shapelets_info = torch.FloatTensor(config['shapelets_info_layer2'])
        self.shapelets_pos_info = torch.index_select(self.shapelets_info, 1, torch.tensor([5, 1, 2]))
        # one-hot pos embedding
        self.shapelets_dim_pos = self.position_embedding(self.shapelets_pos_info[:, 0])
        self.shapelets_start_pos = self.position_embedding(self.shapelets_pos_info[:, 1])
        self.shapelets_end_pos = self.position_embedding(self.shapelets_pos_info[:, 2])

        self.shapelets_dim_pe = nn.Linear(self.shapelets_dim_pos.shape[1], config['shapelets_pos_embed_dim'])
        self.shapelets_start_pe = nn.Linear(self.shapelets_start_pos.shape[1], config['shapelets_pos_embed_dim'])
        self.shapelets_end_pe = nn.Linear(self.shapelets_end_pos.shape[1], config['shapelets_pos_embed_dim'])

        self.specific_LayerNorm1 = nn.LayerNorm(config['shape_embed_dim'], eps=1e-5)
        self.specific_LayerNorm2 = nn.LayerNorm(config['shape_embed_dim'], eps=1e-5)
        self.specific_attention_layer = Attention(config['shape_embed_dim'], config['num_heads'],
                                                  dropout=config['dropout'])

        self.specific_FeedForward = nn.Sequential(
            nn.Linear(config['shape_embed_dim'], config['dim_ff']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['dim_ff'], config['shape_embed_dim']),
            nn.Dropout(config['dropout']))

        self.specific_avgpool = nn.AdaptiveAvgPool1d(1)
        self.specific_flatten = nn.Flatten()

        len_ts = config['len_ts']
        temporal_conv_kernel_size = config['temporal_conv_kernel_size']
        variable_conv_kernel_size = config['variable_conv_kernel_size']

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, config['generic_embed_dim'], kernel_size=[1, temporal_conv_kernel_size], padding='same'),
            nn.BatchNorm2d(config['generic_embed_dim']),
            nn.GELU())

        self.variable_conv = nn.Sequential(
            nn.Conv2d(config['generic_embed_dim'], config['generic_embed_dim'],
                      kernel_size=[variable_conv_kernel_size, 1],
                      stride=[3, 1], padding='valid'),
            nn.BatchNorm2d(config['generic_embed_dim']),
            nn.GELU())

        self.generic_Position_Encoding = LearnablePositionEncoding(config['generic_embed_dim'],
                                                                   dropout=config['dropout'], max_len=len_ts)
        # self.generic_pe_layer = nn.Linear(self.generic_pos_embedding.shape[-1], config['generic_pos_dim'])
        self.generic_LayerNorm1 = nn.LayerNorm(config['generic_embed_dim'], eps=1e-5)
        self.generic_LayerNorm2 = nn.LayerNorm(config['generic_embed_dim'], eps=1e-5)
        self.generic_attention_layer = Attention(config['generic_embed_dim'], config['num_heads'],
                                                 dropout=config['dropout'])
        self.generic_FeedForward = nn.Sequential(
            nn.Linear(config['generic_embed_dim'], config['dim_ff']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['dim_ff'], config['generic_embed_dim']),
            nn.Dropout(config['dropout']))
        self.generic_avgpool = nn.AdaptiveAvgPool1d(1)
        self.generic_flatten = nn.Flatten()

        self.a_out = nn.Linear(config['shape_embed_dim'] + config['generic_embed_dim'],
                               2)  # num_classes = 2 for hierarchical

    def position_embedding(self, position_list):
        max_d = position_list.max() + 1
        identity_matrix = torch.eye(int(max_d))
        d_position = identity_matrix[position_list.to(dtype=torch.long)]
        return d_position

    def forward(self, x):
        specific_x = None
        for block in self.shapeblock:
            if specific_x is None:
                specific_x = block(x)
            else:
                specific_x = torch.cat([specific_x, block(x)], dim=1)

        if self.shapelets_dim_pos.device != x.device:
            self.shapelets_dim_pos = self.shapelets_dim_pos.to(x.device)
            self.shapelets_start_pos = self.shapelets_start_pos.to(x.device)
            self.shapelets_end_pos = self.shapelets_end_pos.to(x.device)

        dim_pos = self.shapelets_dim_pos.repeat(x.shape[0], 1, 1)
        start_pos = self.shapelets_start_pos.repeat(x.shape[0], 1, 1)
        end_pos = self.shapelets_end_pos.repeat(x.shape[0], 1, 1)

        dim_pos_embed = self.shapelets_dim_pe(dim_pos)
        start_pos_embed = self.shapelets_start_pe(start_pos)
        end_pos_embed = self.shapelets_end_pe(end_pos)

        specific_x = specific_x + dim_pos_embed + start_pos_embed + end_pos_embed
        specific_att = specific_x + self.specific_attention_layer(specific_x)
        specific_att = specific_att * self.shape_weight.unsqueeze(0).unsqueeze(2)
        specific_att = self.specific_LayerNorm1(specific_att)
        specific_out = specific_att + self.specific_FeedForward(specific_att)
        specific_out = self.specific_LayerNorm2(specific_out)
        specific_out = specific_out * self.shape_weight.unsqueeze(0).unsqueeze(2)
        specific_out = specific_out[:, 0, :]

        generic_x = x.unsqueeze(1)
        generic_x = self.temporal_conv(generic_x)
        generic_x = self.variable_conv(generic_x).squeeze(2)
        generic_x = generic_x.permute(0, 2, 1)
        generic_x_pe = self.generic_Position_Encoding(generic_x)
        generic_att = generic_x + self.generic_attention_layer(generic_x_pe)
        generic_att = self.generic_LayerNorm1(generic_att)
        generic_out = generic_att + self.generic_FeedForward(generic_att)
        generic_out = self.generic_LayerNorm2(generic_out)
        generic_out = generic_out.permute(0, 2, 1)
        generic_out = self.generic_avgpool(generic_out)
        generic_out = self.generic_flatten(generic_out)

        out = torch.cat([specific_out, generic_out], dim=1)
        out = self.a_out(out)
        return out

class Layer2_GMSA_Net_v1(nn.Module):
    def __init__(self, config):
        super().__init__()
        ks = [[1, 25], [1, 50], [1, 100], [3, 25], [3, 50], [3, 100], [6, 25], [6, 50], [6, 100]]
        self.blocks = nn.ModuleList([GMSA_Block(config, k) for k in ks])
        d = self.blocks[0].out_dim  # 每块输出维度

        self.gate = nn.Sequential(
            nn.LayerNorm(d*len(ks)),
            nn.Linear(d*len(ks), len(ks))
        )
        # 瓶颈+分类头
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, config.get('bottleneck', 512)),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )
        self.out = nn.Linear(config.get('bottleneck', 512), 2)

    def forward(self, x):
        feats = [blk(x) for blk in self.blocks]      # 9 × [B,d]
        F_cat = torch.cat(feats, dim=-1)            # [B, 9d]
        gate = torch.softmax(self.gate(F_cat), dim=-1)  # [B,9]
        F_stack = torch.stack(feats, dim=1)               # [B,9,d]
        Z_mix = (F_stack * gate.unsqueeze(-1)).sum(dim=1) # [B,d] 真正的加权求和
        logits = self.out(self.bottleneck(Z_mix))   # [B,2]
        return logits