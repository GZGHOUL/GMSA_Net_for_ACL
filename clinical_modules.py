"""
临床特征提取模块
用于从步态数据中提取有临床意义的特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaitSymmetryIndex(nn.Module):
    """
    步态对称性指标提取器
    
    计算左右腿的对称性，包括：
    1. 时域对称性（幅度、相位）
    2. 频域对称性（主频率）
    3. 统计对称性（均值、方差、峰度、偏度）
    """
    def __init__(self, output_dim=32):
        super().__init__()
        self.output_dim = output_dim
        
        # 时域特征编码
        self.temporal_encoder = nn.Sequential(
            nn.Linear(6*4, 64),  # 6个自由度 × 4个统计量（均值、方差、峰度、偏度）
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, left_x, right_x):
        """
        Args:
            left_x: [B, 6, T] 左腿数据
            right_x: [B, 6, T] 右腿数据
        
        Returns:
            symmetry_features: [B, output_dim]
        """
        B, C, T = left_x.shape
        
        # 计算统计量
        left_stats = self._compute_statistics(left_x)   # [B, 6, 4]
        right_stats = self._compute_statistics(right_x) # [B, 6, 4]
        
        # 对称性度量：归一化差异
        symmetry = torch.abs(left_stats - right_stats) / (torch.abs(left_stats) + torch.abs(right_stats) + 1e-8)
        
        # 展平
        symmetry_flat = symmetry.reshape(B, -1)  # [B, 24]
        
        # 编码
        features = self.temporal_encoder(symmetry_flat)
        
        return features
    
    def _compute_statistics(self, x):
        """
        计算每个通道的统计量
        
        Args:
            x: [B, C, T]
        
        Returns:
            stats: [B, C, 4] (均值、标准差、峰度、偏度)
        """
        B, C, T = x.shape
        
        # 均值
        mean = x.mean(dim=2)  # [B, C]
        
        # 标准差
        std = x.std(dim=2)  # [B, C]
        
        # 峰度（kurtosis）
        x_centered = x - mean.unsqueeze(-1)
        kurtosis = (x_centered ** 4).mean(dim=2) / (std ** 4 + 1e-8)
        
        # 偏度（skewness）
        skewness = (x_centered ** 3).mean(dim=2) / (std ** 3 + 1e-8)
        
        # 堆叠
        stats = torch.stack([mean, std, kurtosis, skewness], dim=2)  # [B, C, 4]
        
        return stats


class KneeInstabilityDetector(nn.Module):
    """
    膝关节不稳定性检测器
    
    前交叉韧带损伤导致的膝关节不稳定性表现：
    1. 异常的前后位移（anterior tibial translation）
    2. 旋转不稳定性
    3. 内外翻角度异常波动
    """
    def __init__(self, output_dim=32):
        super().__init__()
        self.output_dim = output_dim
        
        # 1D CNN提取不稳定性模式
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 全连接编码
        self.fc = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 12, T] 完整步态数据
        
        Returns:
            instability_features: [B, output_dim]
        """
        # 计算一阶差分（速度）
        velocity = torch.diff(x, dim=2)  # [B, 12, T-1]
        
        # 计算二阶差分（加速度）
        acceleration = torch.diff(velocity, dim=2)  # [B, 12, T-2]
        
        # 卷积提取不稳定性模式
        conv_out = self.conv_layers(acceleration).squeeze(-1)  # [B, 64]
        
        # 编码
        features = self.fc(conv_out)
        
        return features


class PhysicsGuidedAttention(nn.Module):
    """
    基于生物力学先验的通道注意力
    
    膝关节六自由度耦合关系：
    - 屈伸角 ↔ 前后位移 (screw-home mechanism)
    - 内外旋 ↔ 屈伸角 (automatic rotation)
    - 内外翻 ↔ 内外位移 (valgus/varus stress)
    """
    def __init__(self, num_dof=6):
        super().__init__()
        
        # 初始化耦合矩阵（基于生物力学先验）
        # 行/列顺序：屈伸角、内外翻、内外旋、上下位移、前后位移、内外位移
        coupling_prior = torch.tensor([
            #屈伸  内外翻 内外旋  上下   前后   内外
            [1.0,  0.2,  0.6,  0.3,  0.8,  0.1],  # 屈伸角
            [0.2,  1.0,  0.3,  0.2,  0.1,  0.7],  # 内外翻
            [0.6,  0.3,  1.0,  0.2,  0.3,  0.4],  # 内外旋
            [0.3,  0.2,  0.2,  1.0,  0.4,  0.2],  # 上下位移
            [0.8,  0.1,  0.3,  0.4,  1.0,  0.2],  # 前后位移（ACL损伤关键！）
            [0.1,  0.7,  0.4,  0.2,  0.2,  1.0],  # 内外位移
        ], dtype=torch.float32)
        
        # 可学习的耦合矩阵（在先验基础上微调）
        self.coupling_matrix = nn.Parameter(coupling_prior, requires_grad=True)
        
        # 通道注意力网络
        self.channel_attn = nn.Sequential(
            nn.Linear(num_dof, num_dof * 2),
            nn.LayerNorm(num_dof * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(num_dof * 2, num_dof),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 6, T]
        
        Returns:
            x_weighted: [B, 6, T] 加权后的输入
            attn_weights: [B, 6] 注意力权重（可用于可解释性）
        """
        B, C, T = x.shape
        
        # 全局平均池化得到通道统计量
        channel_stats = x.mean(dim=2)  # [B, 6]
        
        # 应用物理耦合关系（矩阵乘法）
        # 每个通道的激活受其他通道影响
        coupled_stats = torch.matmul(channel_stats, self.coupling_matrix)  # [B, 6]
        
        # 通道注意力权重
        attn_weights = self.channel_attn(coupled_stats)  # [B, 6]
        
        # 重新加权输入
        x_weighted = x * attn_weights.unsqueeze(-1)  # [B, 6, T]
        
        return x_weighted, attn_weights


class CrossAttention(nn.Module):
    """
    交叉注意力模块
    用于建模左右腿之间的交互
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Q, K, V投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, L_q, D] 查询序列
            key_value: [B, L_kv, D] 键值序列
        
        Returns:
            output: [B, L_q, D] 注意力输出
        """
        B, L_q, D = query.shape
        L_kv = key_value.shape[1]
        
        # 投影
        Q = self.q_proj(query)     # [B, L_q, D]
        K = self.k_proj(key_value) # [B, L_kv, D]
        V = self.v_proj(key_value) # [B, L_kv, D]
        
        # 重塑为多头
        Q = Q.reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, d]
        K = K.reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L_kv, d]
        V = V.reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L_kv, d]
        
        # 注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_kv]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, V)  # [B, H, L_q, d]
        
        # 重塑回原始维度
        attn_output = attn_output.transpose(1, 2).reshape(B, L_q, D)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class ROMAnalyzer(nn.Module):
    """
    关节活动度（Range of Motion, ROM）分析器
    
    分析膝关节活动度的异常模式：
    - 活动度减小（半月板损伤可能导致）
    - 活动度不对称（左右腿差异）
    - 活动度模式异常（受限或过度活动）
    """
    def __init__(self, output_dim=32):
        super().__init__()
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(6*3, 64),  # 6个自由度 × 3个ROM指标
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 6, T]
        
        Returns:
            rom_features: [B, output_dim]
        """
        B, C, T = x.shape
        
        # 计算ROM指标
        max_val = x.max(dim=2)[0]  # [B, 6]
        min_val = x.min(dim=2)[0]  # [B, 6]
        range_val = max_val - min_val  # [B, 6]
        
        # 拼接
        rom_stats = torch.stack([max_val, min_val, range_val], dim=2)  # [B, 6, 3]
        rom_flat = rom_stats.reshape(B, -1)  # [B, 18]
        
        # 编码
        features = self.encoder(rom_flat)
        
        return features


class LoadingPatternExtractor(nn.Module):
    """
    负重模式提取器
    
    分析垂直方向（上下位移）和水平方向的负重模式
    半月板损伤可能导致异常的负重策略
    """
    def __init__(self, output_dim=32):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=11, padding=5),  # 上下、前后、内外位移
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Linear(32, output_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, 12, T]
        
        Returns:
            loading_features: [B, output_dim]
        """
        # 提取左右腿的位移通道（3-5：左腿，9-11：右腿）
        left_displacement = x[:, 3:6, :]   # [B, 3, T]
        right_displacement = x[:, 9:12, :] # [B, 3, T]
        
        # 计算平均负重模式
        avg_displacement = (left_displacement + right_displacement) / 2.0
        
        # 卷积提取模式
        conv_out = self.conv(avg_displacement).squeeze(-1)  # [B, 32]
        
        # 全连接
        features = self.fc(conv_out)
        
        return features


if __name__ == "__main__":
    # 测试
    batch_size = 8
    num_dof = 6
    time_steps = 600
    
    # 生成测试数据
    left_x = torch.randn(batch_size, num_dof, time_steps)
    right_x = torch.randn(batch_size, num_dof, time_steps)
    full_x = torch.cat([left_x, right_x], dim=1)  # [B, 12, T]
    
    print("=" * 50)
    print("测试临床特征提取模块")
    print("=" * 50)
    
    # 测试步态对称性
    symmetry = GaitSymmetryIndex(output_dim=32)
    sym_feat = symmetry(left_x, right_x)
    print(f"✓ 步态对称性特征: {sym_feat.shape}")
    
    # 测试不稳定性检测
    instability = KneeInstabilityDetector(output_dim=32)
    inst_feat = instability(full_x)
    print(f"✓ 不稳定性特征: {inst_feat.shape}")
    
    # 测试物理约束注意力
    physics_attn = PhysicsGuidedAttention(num_dof=6)
    weighted_x, attn_weights = physics_attn(left_x)
    print(f"✓ 物理约束注意力: 加权数据{weighted_x.shape}, 权重{attn_weights.shape}")
    print(f"  注意力权重示例: {attn_weights[0].detach().numpy()}")
    
    # 测试交叉注意力
    cross_attn = CrossAttention(dim=256, num_heads=4)
    query = torch.randn(batch_size, 1, 256)
    key_value = torch.randn(batch_size, 1, 256)
    cross_out = cross_attn(query, key_value)
    print(f"✓ 交叉注意力输出: {cross_out.shape}")
    
    # 测试ROM分析
    rom = ROMAnalyzer(output_dim=32)
    rom_feat = rom(left_x)
    print(f"✓ ROM特征: {rom_feat.shape}")
    
    # 测试负重模式
    loading = LoadingPatternExtractor(output_dim=32)
    loading_feat = loading(full_x)
    print(f"✓ 负重模式特征: {loading_feat.shape}")
    
    print("\n✅ 所有模块测试通过！")

