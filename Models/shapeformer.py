import numpy as np
from torch import nn
import torch
from Models.Attention import Attention, CrossAttention

class LearnablePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        self.pe = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)
    
    def forward(self, x):
        x = x + self.pe
        return x

class ShapeBlock(nn.Module):
    def __init__(self, shapelet_info=None, shapelet=None, shape_embed_dim=32, len_window_shapeblock=50, len_ts=100, norm=1000, max_ci=3):
        super(ShapeBlock, self).__init__()
        self.dim = shapelet_info[5]
        self.shape_embed_dim = shape_embed_dim
        self.shapelet = nn.Parameter(torch.tensor(shapelet, dtype=torch.float32), requires_grad=True)
        self.len_window_shapeblock = len_window_shapeblock
        self.norm = norm
        self.len_shapelet = shapelet.shape[-1]
        self.weight = shapelet_info[3]
        self.len_ts = len_ts

        self.shapelet_ci = np.sqrt(np.sum((shapelet[1:] - shapelet[:-1])**2) + 1/self.norm)
        self.max_ci = max_ci

        self.start_pos = int(shapelet_info[1] - self.len_window_shapeblock)
        self.start_pos = self.start_pos if self.start_pos >= 0 else 0
        self.end_pos = int(shapelet_info[2] + self.len_window_shapeblock)
        self.end_pos = self.end_pos if self.end_pos < self.len_ts else self.len_ts

        self.l1 = nn.Linear(self.len_shapelet, self.shape_embed_dim)
        self.l2 = nn.Linear(self.len_shapelet, self.shape_embed_dim)

    
    def forward(self, x):
        piss = x[:, self.dim, self.start_pos:self.end_pos] # shape:[batch_size, len_shapelet]
        piss_ci = torch.square(torch.subtract(piss[:, 1:], piss[:, :-1]))

        piss = piss.unfold(1, self.len_shapelet, 1).contiguous()
        piss = piss.view(-1, self.len_shapelet)

        piss_ci = piss_ci.unfold(1, self.len_shapelet - 1, 1).contiguous()
        piss_ci = piss_ci.view(-1, self.len_shapelet - 1)
        piss_ci = torch.sum(piss_ci, dim=1) + (1/self.norm)
        piss_ci = torch.sqrt(piss_ci)

        shapelet_ci = torch.ones(piss_ci.size(0), device=x.device, requires_grad=False) * self.shapelet_ci
        max_ci = torch.max(piss_ci, shapelet_ci)
        min_ci = torch.min(piss_ci, shapelet_ci)
        CF = max_ci/min_ci
        CF[CF > self.max_ci] = self.max_ci
        ED = torch.sqrt(torch.sum(torch.square(piss - self.shapelet), dim=1))
        CID = ED * CF
        CID = CID.view(x.size(0), -1)

        index = torch.argmin(CID, dim=1)
        piss = piss.view(x.size(0), -1, self.len_shapelet)
        out_i = piss[torch.arange(int(x.size(0))).to(torch.long).to(x.device), index.to(torch.long)]
        out_i = self.l1(out_i)

        out_s = self.l2(self.shapelet.unsqueeze(0))

        out = out_i - out_s

        return out.view(x.size(0), 1, -1)

class ShapeFormer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.use_b_branch = config['use_b_branch']

        # 存储形状子信息（【样本索引， 起始帧， 终止帧， 信息增益权重， 类别， 维度】）和形状子序列
        self.shapelets_info = config['shapelets_info']
        self.shapelets_info = torch.IntTensor(self.shapelets_info)
        self.shapelets = config['shapelets']

        # 形状子权重参数（可学习）：基于IG初始化，训练中动态调整形状子重要性
        self.shape_weight = torch.nn.Parameter(torch.tensor(config['shapelets_info'][:, 3]).float(), requires_grad=True)

        # Generic Transformer

        # 窗口分割参数：将长时间序列切分为固定长度窗口
        self.len_window_generic_transformer = config['len_window_generic_transformer']
        self.pad_w = self.len_window_generic_transformer - config['len_ts'] % self.len_window_generic_transformer # 填充长度，确保最后一个窗口完整
        self.pad_w = 0 if self.pad_w == self.len_window_generic_transformer else self.pad_w
        self.height = config['ts_dim']
        self.width = int(np.ceil(config['len_ts']/self.len_window_generic_transformer))

        # 局部位置嵌入：生成每个窗口的“维度+位置”嵌入，捕捉窗口的空间（维度）和时序（位置）信息
        list_dim = []
        list_pos = []
        for dim in range(self.height):
            for pos in range(self.width):
                list_dim.append(dim)
                list_pos.append(pos)

        list_dim_embedding = self.position_embedding(torch.tensor(list_dim)) # 维度位置嵌入
        list_pos_embedding = self.position_embedding(torch.tensor(list_pos)) # 窗口位置嵌入
        self.generic_pos_embedding = torch.cat([list_dim_embedding, list_pos_embedding], dim=1) # 合并为（维度+位置）嵌入

        len_ts = config['len_ts']
        temporal_conv_kernel_size = config['temporal_conv_kernel_size']
        variable_conv_kernel_size = config['ts_dim']

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, config['generic_embed_dim'], kernel_size=[1, temporal_conv_kernel_size], padding='same'),
            nn.BatchNorm2d(config['generic_embed_dim']),
            nn.GELU())

        self.variable_conv = nn.Sequential(
            nn.Conv2d(config['generic_embed_dim'], config['generic_embed_dim'], kernel_size=[variable_conv_kernel_size, 1], padding='valid'),
            nn.BatchNorm2d(config['generic_embed_dim']),
            nn.GELU())
        
        self.generic_Position_Encoding = LearnablePositionEncoding(config['generic_embed_dim'], dropout=config['dropout'], max_len=len_ts)
        # self.generic_pe_layer = nn.Linear(self.generic_pos_embedding.shape[-1], config['generic_pos_dim'])
        self.generic_LayerNorm1 = nn.LayerNorm(config['generic_embed_dim'], eps=1e-5)
        self.generic_LayerNorm2 = nn.LayerNorm(config['generic_embed_dim'], eps=1e-5)
        self.generic_attention_layer = Attention(config['generic_embed_dim'], config['num_heads'], dropout=config['dropout'])
        self.generic_FeedForward = nn.Sequential(
            nn.Linear(config['generic_embed_dim'], config['dim_ff']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['dim_ff'], config['generic_embed_dim']),
            nn.Dropout(config['dropout']))
        self.generic_avgpool = nn.AdaptiveAvgPool1d(1)
        self.generic_flatten = nn.Flatten()

        # Class-specific Transformer
        self.shapeblock = nn.ModuleList([
            ShapeBlock(shapelet_info=self.shapelets_info[i], shapelet=self.shapelets[i], shape_embed_dim=config['shape_embed_dim'],
            len_window_shapeblock=config['len_window_shapeblock'], len_ts=config['len_ts'], norm=config['norm'], max_ci=config['max_ci'])
            for i in range(len(self.shapelets_info))
        ])

        self.shapelets_info = torch.FloatTensor(config['shapelets_info'])
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
        self.specific_attention_layer = Attention(config['shape_embed_dim'], config['num_heads'], dropout=config['dropout'])

        self.specific_FeedForward = nn.Sequential(
            nn.Linear(config['shape_embed_dim'], config['dim_ff']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['dim_ff'], config['shape_embed_dim']),
            nn.Dropout(config['dropout']))

        self.specific_avgpool = nn.AdaptiveAvgPool1d(1)
        self.specific_flatten = nn.Flatten()
        self.a_out = nn.Linear(config['shape_embed_dim'] + config['generic_embed_dim'], num_classes)

        # B branch (period - normalized 12*100)
        b_temporal_conv_kernel_size = config['b_temporal_conv_kernel_size']
        b_variable_conv_kernel_size = config['b_variable_conv_kernel_size']
        len_ts_b = config['len_ts_b']
        
        self.b_temporal_conv = nn.Sequential(
            nn.Conv2d(1, config['b_embed_dim'], kernel_size=[1, b_temporal_conv_kernel_size], padding='same'),
            nn.BatchNorm2d(config['b_embed_dim']),
            nn.GELU())
        self.b_variable_conv = nn.Sequential(
            nn.Conv2d(config['b_embed_dim'], config['b_embed_dim'], kernel_size=[b_variable_conv_kernel_size, 1], padding='valid'),
            nn.BatchNorm2d(config['b_embed_dim']),
            nn.GELU())
        self.b_Position_Encoding = LearnablePositionEncoding(config['b_embed_dim'], dropout=config['dropout'], max_len=len_ts_b)
        self.b_LayerNorm1 = nn.LayerNorm(config['b_embed_dim'], eps=1e-5)
        self.b_LayerNorm2 = nn.LayerNorm(config['b_embed_dim'], eps=1e-5)
        self.b_cross_attention_lr = CrossAttention(config['b_embed_dim'], config['num_heads'], dropout=config['dropout'])
        self.b_cross_attention_rl = CrossAttention(config['b_embed_dim'], config['num_heads'], dropout=config['dropout'])
        self.b_FeedForward = nn.Sequential(
            nn.Linear(config['b_embed_dim'], config['dim_ff']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['dim_ff'], config['b_embed_dim']),
            nn.Dropout(config['dropout']))
        self.b_avgpool = nn.AdaptiveAvgPool1d(1)
        self.b_flatten = nn.Flatten()
        self.b_out = nn.Linear(config['b_embed_dim'] * 2, config['generic_embed_dim'])

        # self.out = nn.Linear(config['shape_embed_dim'] + config['generic_embed_dim'] + config['b_embed_dim'] * 2, num_classes)
        self.out = nn.Linear(config['shape_embed_dim'] + config['generic_embed_dim'] + config['generic_embed_dim'], num_classes)


    def position_embedding(self, position_list):
        max_d = position_list.max() + 1
        identity_matrix = torch.eye(int(max_d))
        d_position = identity_matrix[position_list.to(dtype=torch.long)]
        return d_position

    def forward(self, x):
        if self.use_b_branch and isinstance(x, (tuple, list)):
            x_a, x_b = x
        else:
            x_a, x_b = x, None
        
        generic_x = x_a.unsqueeze(1)
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

        specific_x = None
        for block in self.shapeblock:
            if specific_x is None:
                specific_x = block(x_a)
            else:
                specific_x = torch.cat([specific_x, block(x_a)], dim=1)

        if self.shapelets_dim_pos.device != x_a.device:
            self.shapelets_dim_pos = self.shapelets_dim_pos.to(x_a.device)
            self.shapelets_start_pos = self.shapelets_start_pos.to(x_a.device)
            self.shapelets_end_pos = self.shapelets_end_pos.to(x_a.device)

        dim_pos = self.shapelets_dim_pos.repeat(x_a.shape[0], 1, 1)
        start_pos = self.shapelets_start_pos.repeat(x_a.shape[0], 1, 1)
        end_pos = self.shapelets_end_pos.repeat(x_a.shape[0], 1, 1)

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

        if x_b is not None:
            left_leg = x_b[:, :6, :]
            right_leg = x_b[:, 6:, :]
            left_leg = left_leg.unsqueeze(1)
            right_leg = right_leg.unsqueeze(1)

            left_leg = self.b_temporal_conv(left_leg)
            left_leg = self.b_variable_conv(left_leg).squeeze(2)
            left_leg = left_leg.permute(0, 2, 1)

            right_leg = self.b_temporal_conv(right_leg)
            right_leg = self.b_variable_conv(right_leg).squeeze(2)
            right_leg = right_leg.permute(0, 2, 1)

            left_leg_pe = self.b_Position_Encoding(left_leg)
            right_leg_pe = self.b_Position_Encoding(right_leg)

            lr_cross_att = left_leg + self.b_cross_attention_lr(left_leg_pe, s=right_leg_pe)
            rl_cross_att = right_leg + self.b_cross_attention_rl(right_leg_pe, s=left_leg_pe)

            lr_cross_att = self.b_LayerNorm1(lr_cross_att)
            rl_cross_att = self.b_LayerNorm1(rl_cross_att)

            lr_cross_out = lr_cross_att + self.b_FeedForward(lr_cross_att)
            rl_cross_out = rl_cross_att + self.b_FeedForward(rl_cross_att)

            lr_cross_out = self.b_LayerNorm2(lr_cross_out)
            rl_cross_out = self.b_LayerNorm2(rl_cross_out)
            lr_cross_out = lr_cross_out.permute(0, 2, 1)
            rl_cross_out = rl_cross_out.permute(0, 2, 1)

            lr_cross_out = self.b_avgpool(lr_cross_out)
            rl_cross_out = self.b_avgpool(rl_cross_out)
            lr_cross_out = self.b_flatten(lr_cross_out)
            rl_cross_out = self.b_flatten(rl_cross_out)
            b_out = self.b_out(torch.cat([lr_cross_out, rl_cross_out], dim=-1))
        else:
            b_out = torch.zeros_like(generic_out)

        out = torch.cat([specific_out, generic_out, b_out], dim=1)
        out = self.out(out)

        return out



            


        
        







