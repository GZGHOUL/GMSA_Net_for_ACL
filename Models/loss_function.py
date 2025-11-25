import torch
import torch.nn as nn
from torch.nn import functional as F

def get_loss_function(loss_function='CE', device=None):
    # 1. 定义类别权重 (根据之前的讨论: 健康=1.0, 左患=1.5, 右患=1.2)
    # 务必确保权重在正确的 device 上
    class_weights = torch.tensor([1.0, 1.5, 1.2]).to(device)

    if loss_function == 'CE':
        # 修正1: 传入 weight 参数
        # 修正2: reduction 改为 'mean' 以便直接 backward
        return NoFussCrossEntropyLoss(weight=class_weights, reduction='none')
    
    elif loss_function == 'FocalLoss':
        # Focal Loss 的 alpha 策略 (之前讨论的强力纠偏: [1.0, 4.0, 2.0])
        focal_alpha = torch.tensor([1.0, 4.0, 2.0]).to(device)
        # 修正: reduction 改为 'mean'
        return FocalLoss(alpha=focal_alpha, gamma=2.0, reduction='none')

def l2_reg_loss(model):
    """L2正则,对所有可训练的权重参数进行L2惩罚(排除偏置与标量参数)"""
    # 这里的逻辑是没问题的，用于手动添加 L2 Loss
    # 注意：如果你使用了 AdamW 优化器并设置了 weight_decay，这个函数其实是多余的
    total = torch.tensor(0., device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if(param.requires_grad and param.dim() > 1):
            total = total + torch.sum(param.pow(2))
    return total

class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    封装的 CE Loss，强制开启 Label Smoothing
    """
    def __init__(self, weight=None, reduction='mean', label_smoothing=0.1):
        # 必须显式调用父类初始化，传入 weight
        super().__init__(weight=weight, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, input, target):
        # 直接使用父类的 forward 即可，参数已经在 __init__ 里传给父类了
        # 这样更简洁，且利用了 PyTorch 底层优化
        return super().forward(input, target.long())

class FocalLoss(nn.Module):
    """Focal Loss 专门针对类别不平衡 (拼写已修正)""" 
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha   # 类别权重
        self.gamma = gamma   # 聚焦参数
        self.reduction = reduction
     
    def forward(self, inputs, targets):
        # 1. 计算标准 CE (不缩减，为了后面乘系数)
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        
        # 2. 计算 pt
        pt = torch.exp(-ce_loss)
        
        # 3. 计算 Focal Term
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. 应用 Alpha (类别权重)
        if self.alpha is not None:
            # 确保 alpha 和 targets 在同一设备
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets.long()]
            focal_loss = alpha_t * focal_loss
        
        # 5. 根据 reduction 返回结果
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss