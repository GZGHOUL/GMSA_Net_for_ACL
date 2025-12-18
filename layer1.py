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

        self.use_diff = config.get('use_diff_channel', True)

        # ========== 只保留3个关键尺度 ==========
        self.ks = [
            [1, 5], 
            [3, 20],
            [6, 40]
        ]
        
        leg_config = self._adapt_config_for_single_leg(config)
        
        # 1. 共享特征提取器 (Left/Right) - 始终需要
        self.shared_blocks = nn.ModuleList([
            GMSA_Block(leg_config, k) for k in self.ks
        ])
        
        d = self.shared_blocks[0].out_dim
        num_blocks = len(self.ks)
        
        self.left_gate = self._make_gate(d, num_blocks)
        self.right_gate = self._make_gate(d, num_blocks)

        # 2. 动态构建 Diff 相关模块
        if self.use_diff:
            # 开启时：需要 Diff 流和 AbsDiff 投影
            self.diff_blocks = nn.ModuleList([
                GMSA_Block(leg_config, k) for k in self.ks
            ])
            self.diff_gate = self._make_gate(d, num_blocks)
            
            self.abs_diff_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(6 * 100, d), 
                nn.LayerNorm(d),
                nn.GELU()
            )
            # 融合维度: L + R + Diff + FeatDiff + AbsDiff = 5d
            fusion_input_dim = d * 5
        else:
            # 关闭时：不需要这些参数
            self.diff_blocks = None
            self.diff_gate = None
            self.abs_diff_proj = None
            # 融合维度: L + R + FeatDiff = 3d
            fusion_input_dim = d * 3

        # 3. 融合层与分类头
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256), 
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )
        
        # 动态输出维度 (2分类或3分类)
        num_classes = 3 if config.get('layer1_task') == 'ternary' else 2
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )
    
    def _adapt_config_for_single_leg(self, config):
        adapted_config = config.copy()
        adapted_config['ts_dim'] = 6
        adapted_config['len_ts'] = config.get('len_ts', 100)
        return adapted_config
    
    def _make_gate(self, d, num_blocks):
        return nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Dropout(0.2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x, return_features=False):
        # 1. 基础切片 (Left/Right)
        # 无论有没有 Diff，前12个通道总是 L(0-5) 和 R(6-11)
        left_x = x[:, :6, :]      
        right_x = x[:, 6:12, :]   
        
        # 2. 基础特征提取
        left_feats = [blk(left_x) for blk in self.shared_blocks]
        right_feats = [blk(right_x) for blk in self.shared_blocks]
        
        # 3. Gating
        left_fused = self._apply_gate(left_feats, self.left_gate)
        right_fused = self._apply_gate(right_feats, self.right_gate)
        
        # 4. 特征级差分 (始终保留)
        feat_difference = left_fused - right_fused 
        
        # 5. 分支处理：是否使用 Diff 通道
        if self.use_diff:
            # 输入 Shape: (B, 24, T)
            diff_x = x[:, 12:18, :]   
            abs_diff_x = x[:, 18:, :] 
            
            # Diff 流提取
            diff_feats = [blk(diff_x) for blk in self.diff_blocks]
            diff_fused = self._apply_gate(diff_feats, self.diff_gate)
            
            # AbsDiff 投影
            abs_diff_feat = self.abs_diff_proj(abs_diff_x)
            
            # 融合: 5部分
            combined = torch.cat([
                left_fused, 
                right_fused, 
                diff_fused * 1.5, 
                feat_difference, 
                abs_diff_feat
            ], dim=-1)
            
        else:
            # 输入 Shape: (B, 12, T)
            # 融合: 3部分 (仅使用纯净的原始数据)
            combined = torch.cat([
                left_fused, 
                right_fused, 
                feat_difference
            ], dim=-1)
        
        features = self.feature_fusion(combined)
        logits = self.classifier(features)
        
        if return_features:
            return logits, {'features': features}
        
        return logits

    def _apply_gate(self, feat_list, gate_layer):
        cat = torch.cat(feat_list, dim=-1)
        weights = gate_layer(cat)
        stack = torch.stack(feat_list, dim=1)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return fused


class Layer1_AsyncDualStream_Lite_clinical(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_diff = config.get('use_diff_channel', True)

        # ========== 只保留3个关键尺度 ==========
        self.ks = [
            [1, 5],
            [3, 20],
            [6, 40]
        ]

        leg_config = self._adapt_config_for_single_leg(config)

        # 1. 共享特征提取器 (Left/Right) - 始终需要
        self.shared_blocks = nn.ModuleList([
            GMSA_Block(leg_config, k) for k in self.ks
        ])

        d = self.shared_blocks[0].out_dim
        num_blocks = len(self.ks)

        self.left_gate = self._make_gate(d, num_blocks)
        self.right_gate = self._make_gate(d, num_blocks)

        # [新增] 动力学特征投影层 (Dynamics Projection)
        # 输入: 12个通道 (L_acc + R_acc) * 长度 (len_ts)
        # 我们用一个简单的 MLP 将其压缩到 d 维
        # 注意: 需要从 config 获取 len_ts，如果未定义默认 100
        ts_len = config.get('len_ts', 100)
        self.dynamics_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * ts_len, d * 2),  # 稍微宽一点，保留更多信息
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(d * 2, d)
        )

        # 2. 动态构建 Diff 相关模块
        if self.use_diff:
            # 开启时：需要 Diff 流和 AbsDiff 投影
            self.diff_blocks = nn.ModuleList([
                GMSA_Block(leg_config, k) for k in self.ks
            ])
            self.diff_gate = self._make_gate(d, num_blocks)

            self.abs_diff_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(6 * 100, d),
                nn.LayerNorm(d),
                nn.GELU()
            )
            fusion_input_dim = d * 8
        else:
            # 关闭时：不需要这些参数
            self.diff_blocks = None
            self.diff_gate = None
            self.abs_diff_proj = None

            fusion_input_dim = d * 6

        # 3. 融合层 (Feature Fusion)
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.6),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.6),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        num_classes = 3

        # 临床特征数量 (L_rom, R_rom, Diff_rom, L_peak, R_peak, Diff_peak)
        self.num_clinical_features = 6

        # 输入维度 = 神经网络特征(64) + 临床特征(6) = 70
        self.classifier = nn.Sequential(
            nn.Linear(64 + self.num_clinical_features, num_classes)
        )

    def _adapt_config_for_single_leg(self, config):
        adapted_config = config.copy()
        adapted_config['ts_dim'] = 6
        adapted_config['len_ts'] = config.get('len_ts', 100)
        return adapted_config

    def _make_gate(self, d, num_blocks):
        return nn.Sequential(
            nn.LayerNorm(d * num_blocks),
            nn.Linear(d * num_blocks, num_blocks),
            nn.Dropout(0.2),
            nn.Softmax(dim=-1)
        )

    def compute_clinical_features(self, x):
        """
        计算显式的临床物理特征
        x: [B, C, T]
        """
        # 假设前6是左(0-5), 后6是右(6-11)
        # 通道2是屈伸角 (Flexion) -> Left: 2, Right: 8
        L_flex = x[:, 2, :]
        R_flex = x[:, 8, :]

        # 1. ROM (Range of Motion)
        # max/min return (values, indices) -> 取 [0]
        L_rom = L_flex.max(dim=1)[0] - L_flex.min(dim=1)[0]
        R_rom = R_flex.max(dim=1)[0] - R_flex.min(dim=1)[0]
        Diff_rom = (L_rom - R_rom).abs()

        # 2. 支撑相峰值 (Stance Peak) - 假设前40%是支撑相
        # 注意：这里需要确保 T >= 40，否则切片报错。
        t_len = x.shape[2]
        t_40 = int(t_len * 0.4)

        L_peak = L_flex[:, :t_40].max(dim=1)[0]
        R_peak = R_flex[:, :t_40].max(dim=1)[0]
        Diff_peak = (L_peak - R_peak).abs()

        # 拼接成特征向量 [B, 6]
        return torch.stack([L_rom, R_rom, Diff_rom, L_peak, R_peak, Diff_peak], dim=1)

    def forward(self, x, return_features=False):
        # 基础切片
        left_x = x[:, :6, :]
        right_x = x[:, 6:12, :]

        # GMSA 特征提取 (针对位置数据)
        left_feats = [blk(left_x) for blk in self.shared_blocks]
        right_feats = [blk(right_x) for blk in self.shared_blocks]

        # Gating
        left_fused = self._apply_gate(left_feats, self.left_gate)
        right_fused = self._apply_gate(right_feats, self.right_gate)

        # 特征级差分
        feat_difference = left_fused - right_fused

        # 处理新增的动力学特征 (Dynamics)
        # 取出 L_acc 和 R_acc
        if x.shape[1] >= 36:
            l_acc = x[:, 24:30, :]  # 取 12 个通道
            r_acc = x[:, 30, 36, :]
            dynamics_feat_l = self.dynamics_proj(l_acc)  # [B, d]
            dynamics_feat_r = self.dynamics_proj(r_acc)
            dynamics_difference = dynamics_feat_l - dynamics_feat_r
        else:
            # 兼容旧数据防止报错 (如果 config 没开 add_diff_channel)
            dynamics_feat_l = torch.zeros_like(left_fused)
            dynamics_feat_r = torch.zeros_like(left_fused)
            dynamics_difference = torch.zeros_like(left_fused)

        # 5. 分支处理：是否使用 Diff 通道
        if self.use_diff:
            diff_x = x[:, 12:18, :]
            abs_diff_x = x[:, 18:24, :]

            diff_feats = [blk(diff_x) for blk in self.diff_blocks]
            diff_fused = self._apply_gate(diff_feats, self.diff_gate)
            abs_diff_feat = self.abs_diff_proj(abs_diff_x)

            # 全局融合
            combined = torch.cat([
                left_fused,
                right_fused,
                diff_fused * 3.0,
                feat_difference,
                abs_diff_feat,
                dynamics_feat_l * 2.0,
                dynamics_feat_r * 2.0,
                dynamics_difference * 2.0
            ], dim=-1)

        else:
            combined = torch.cat([
                left_fused,
                right_fused,
                feat_difference,
                dynamics_feat_l,
                dynamics_feat_r,
                dynamics_difference
            ], dim=-1)

        # 深层网络特征 [B, 64]
        features = self.feature_fusion(combined)

        # 6. [新增] 计算并融合临床特征 (Late Fusion)
        clinical_feat = self.compute_clinical_features(x)

        # 拼接：[B, 64] + [B, 6] -> [B, 70]
        # 给临床特征 * 10.0 的权重，强迫分类器重视物理指标
        features_final = torch.cat([features, clinical_feat * 1.0], dim=1)

        # 分类
        logits = self.classifier(features_final)

        if return_features:
            return logits, {'features': features_final}

        return logits

    def _apply_gate(self, feat_list, gate_layer):
        cat = torch.cat(feat_list, dim=-1)
        weights = gate_layer(cat)
        stack = torch.stack(feat_list, dim=1)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return fused
