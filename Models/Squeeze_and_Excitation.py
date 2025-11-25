from torch import nn

class SE_Block_1D(nn.Module):
    def __init__(self, input_dim, reduction=4):
        """
        Args:
            input_dim: 输入特征的维度 (例如 32*3 = 96)
            reduction: 降维比例，越小参数越多，计算越精细。
                       由于你的特征维度不大(约96)，建议设为 4 或 8，不要设太大(16)。
        """
        super().__init__()
        self.fc = nn.Sequential(
            # 1. Squeeze: 降维 (压缩信息)
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(),
            # 2. Excitation: 升维 (恢复维度)
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            # 3. Sigmoid: 输出 0~1 之间的权重
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, input_dim)
        
        # 计算每个通道的权重 (Attention Weights)
        weights = self.fc(x)
        
        # 将权重乘回原特征 (Rescale)
        return x * weights