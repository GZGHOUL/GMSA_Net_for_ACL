import torch
import torch.nn as nn
from Models.Attention import Attention

class GMSA_Block(nn.Module):
    def __init__(self, config, kernel_size):
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        self.stride = [self.kernel_size[0], 2]
        # self.embed_dim = self.kernel_size[0] * self.kernel_size[1]
        self.embed_dim = config['embed_dim']
        self.num_patch = (config['ts_dim']//self.kernel_size[0]) * ((config['len_ts'] - self.kernel_size[1])//self.stride[1] + 1)
        
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.embed_dim, kernel_size=self.kernel_size, stride=self.stride, padding='valid'),
            nn.GroupNorm(num_groups=4, num_channels=self.embed_dim),
            nn.GELU())

        self.flatten = nn.Flatten()

        self.position_embedding = LearnablePositionEncoding(self.embed_dim, max_len=self.num_patch)
        self.attention_layer = Attention(self.embed_dim, 4, dropout=config['dropout'])

        self.ln1 = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.FeedForward = nn.Sequential(
            nn.Linear(self.embed_dim, config['dim_ff']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['dim_ff'], self.embed_dim),
            nn.Dropout(config['dropout']))

        self.ln2 = nn.LayerNorm(self.embed_dim, eps=1e-5)

        d_out = config.get('head_dim', 32)
        self.pool = AttnPool1D(self.embed_dim)
        self.reduce = nn.Linear(self.embed_dim, d_out)
        self.out_dim = d_out

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x).permute(0, 3, 2, 1).flatten(start_dim=1, end_dim=2)
        x = self.position_embedding(x)
        att_out = x + self.attention_layer(self.ln1(x))
        att_out = att_out + self.FeedForward(self.ln2(att_out))
        att_out = self.pool(att_out)
        att_out = self.reduce(att_out)

        return att_out

class AttnPool1D(nn.Module):
    def __init__(self, d):
        super().__init__()
        # 可学习查询向量（做全局汇聚用）
        self.q = nn.Parameter(torch.randn(1, 1, d))
        self.proj = nn.Linear(d, d)
    
    def forward(self, x):
        q = self.q.expand(x.size(0), -1, -1)
        attn = torch.softmax((torch.matmul(q, x.transpose(1, 2)) / (x.size(-1) ** 0.5)), dim=-1)
        pooled = torch.matmul(attn, x).squeeze(1)
        out = self.proj(pooled)
        return out

class LearnablePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        self.pe = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)
    
    def forward(self, x):
        x = x + self.pe
        return x
