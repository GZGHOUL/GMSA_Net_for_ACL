import torch
import torch.nn as nn
from Models.Attention import Attention, CrossAttention
from Models.shapeformer import ShapeBlock, LearnablePositionEncoding
from Models.Global_Multi_scale_Attention import GMSA_Block
from Models.Squeeze_and_Excitation import SE_Block_1D

class Layer1_shapeformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 存储形状子信息（【样本索引， 起始帧， 终止帧， 信息增益权重， 类别， 维度】）和形状子序列
        self.shapelets_info = config['shapelets_info_layer1']
        self.shapelets_info = torch.IntTensor(self.shapelets_info)
        self.shapelets = config['shapelets_layer1']

        # 形状子权重参数（可学习）：基于IG初始化，训练中动态调整形状子重要性
        self.shape_weight = torch.nn.Parameter(torch.tensor(config['shapelets_info_layer1'][:, 3]).float(),
                                               requires_grad=True)

        # Class-specific Transformer
        self.shapeblock = nn.ModuleList([
            ShapeBlock(shapelet_info=self.shapelets_info[i], shapelet=self.shapelets[i],
                       shape_embed_dim=config['shape_embed_dim'],
                       len_window_shapeblock=config['len_window_shapeblock'], len_ts=config['len_ts'],
                       norm=config['norm'], max_ci=config['max_ci'])
            for i in range(len(self.shapelets_info))
        ])

        self.shapelets_info = torch.FloatTensor(config['shapelets_info_layer1'])
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

class Layer1_AsyncDualStream_Lite(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # ========== 只保留3个关键尺度 ==========
        self.ks = [
            [1,5],   # 单通道，中等时间窗
            [3,20],   # 3通道组合
            [6,40]    # 整条腿
        ]
        
        # 更新 config 中的时间序列长度，防止 GMSA 计算 patch数出错
        leg_config = self._adapt_config_for_single_leg(config)
        
        # self.left_blocks = nn.ModuleList([
        #     GMSA_Block(leg_config, k) for k in self.ks
        # ])
        
        # self.right_blocks = nn.ModuleList([
        #     GMSA_Block(leg_config, k) for k in self.ks
        # ])
        
        self.shared_blocks = nn.ModuleList([
            GMSA_Block(leg_config, k) for k in self.ks
        ])

        # 新增：差值流 (专门学习不对称性)
        self.diff_blocks = nn.ModuleList([
            GMSA_Block(leg_config, k) for k in self.ks
        ])
        
        d = self.shared_blocks[0].out_dim
        num_blocks = len(self.ks)
        
        self.left_gate = self._make_gate(d, num_blocks)
        self.right_gate = self._make_gate(d, num_blocks)
        self.diff_gate = self._make_gate(d, num_blocks) # 新增

        self.se_block = SE_Block_1D(input_dim=(d * 3), reduction=4)

        self.feature_fusion = nn.Sequential(
            nn.Linear(d * 3, 128), 
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.6), 
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.6)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(32, 3) # 三分类
        )
    
    def _adapt_config_for_single_leg(self, config):
        adapted_config = config.copy()
        adapted_config['ts_dim'] = 6
        adapted_config['len_ts'] = 100 
        return adapted_config
    
    def _make_gate(self, d, num_blocks):
        return nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Dropout(0.2), 
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x, return_features=False):
        # x shape: (Batch, 18, 100)
        
        left_x = x[:, :6, :]      # Channel 0-5
        right_x = x[:, 6:12, :]   # Channel 6-11
        diff_x = x[:, 12:, :]     # Channel 12-17 (Diff)
        
        # --- 1. 特征提取 ---
        left_feats = [blk(left_x) for blk in self.shared_blocks]
        right_feats = [blk(right_x) for blk in self.shared_blocks]
        diff_feats = [blk(diff_x) for blk in self.diff_blocks]
        
        # --- 2. Gating 融合 (加权求和) ---
        left_fused = self._apply_gate(left_feats, self.left_gate)
        right_fused = self._apply_gate(right_feats, self.right_gate)
        diff_fused = self._apply_gate(diff_feats, self.diff_gate)
        diff_fused = diff_fused * 2.0
        
        # --- 3. 全局融合 ---
        # 拼接: (B, d) * 3 -> (B, 3d)
        combined = torch.cat([left_fused, right_fused, diff_fused], dim=-1)
        # combined = self.se_block(combined)
        features = self.feature_fusion(combined)
        
        # --- 4. 分类 ---
        logits = self.classifier(features)
        
        if return_features:
            return logits, {
                'left': left_fused,
                'right': right_fused,
                'diff': diff_fused,
                'fused': features
            }
        
        return logits

    def _apply_gate(self, feat_list, gate_layer):
        """辅助函数: 应用门控机制"""
        # feat_list: [ (B, d), (B, d), (B, d) ]
        cat = torch.cat(feat_list, dim=-1)      # (B, 3*d)
        weights = gate_layer(cat)               # (B, 3)
        stack = torch.stack(feat_list, dim=1)   # (B, 3, d)
        # 加权求和: sum( (B,3,d) * (B,3,1) ) -> (B, d)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return fused


class Layer1_leg_v2(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # ========== 只保留3个关键尺度 ==========
        self.ks = [
            [1,100],   # 单通道，中等时间窗
            [3,100],   # 3通道组合
            [6,100]    # 整条腿
        ]
        
        # 单腿腿GMSA（减少到3个）
        self.leg_blocks = nn.ModuleList([
            GMSA_Block(self._adapt_config_for_single_leg(config), k) 
            for k in self.ks
        ])
        
        d = self.leg_blocks[0].out_dim
        num_blocks = len(self.ks)
        
        # ========== 简化的Gating ==========
        self.leg_gate = nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Dropout(0.3),  # Gating也加dropout
            nn.Softmax(dim=-1)
        )
        
        # ========== 简化的特征投影 ==========
        self.feature_fusion = nn.Sequential(
            nn.Linear(d, 256),  # 左+右 → 256
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
    
    def forward(self, x):
        # GMSA特征提取
        leg_feats = [blk(x) for blk in self.leg_blocks]
        
        # Gating融合
        leg_cat = torch.cat(leg_feats, dim=-1)
        leg_gate_weights = self.leg_gate(leg_cat)
        leg_stack = torch.stack(leg_feats, dim=1)
        leg_fused = (leg_stack * leg_gate_weights.unsqueeze(-1)).sum(dim=1)
        
        features = self.feature_fusion(leg_fused)
        # 单腿分类
        leg_logits = self.classifier(features)
        
        return leg_logits