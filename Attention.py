import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = embed_dim ** -0.5

        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(embed_dim)
        self.att = None

    def forward(self, x):

        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        att = torch.matmul(q, k) * self.scale
        att = nn.functional.softmax(att, dim=-1)

        self.att = att

        out = torch.matmul(att, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)

        out = self.to_out(out)

        return out

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = embed_dim ** -0.5

        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(embed_dim)

    def forward(self, x, s=None):

        batch_size, seq_len, _ = x.shape
        k = self.key(s).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(s).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        att = torch.matmul(q, k) * self.scale
        att = nn.functional.softmax(att, dim=-1)

        out = torch.matmul(att, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)

        out = self.to_out(out)

        return out
