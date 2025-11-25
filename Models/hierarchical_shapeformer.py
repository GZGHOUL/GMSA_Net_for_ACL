import torch
import torch.nn as nn
from Models.layer1 import Layer1_shapeformer, Layer1_AsyncDualStream_Lite, Layer1_leg_v2
from Models.layer2 import Layer2_AsyncDualStream_Lite

class hierarchical_shapeformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer1 = Layer1_AsyncDualStream_Lite(config)
        self.layer2 = Layer2_AsyncDualStream_Lite(config)
    
    def forward(self, x, mask=None, layer='both'):
        if layer == 'layer1':
            return self.layer1(x)
        elif layer == 'layer2':
            return self.layer2(x)
        elif layer == 'both' and mask != None:
            # 第一层预测
            layer1_logits = self.layer1(x)
            layer1_probs = torch.softmax(layer1_logits, dim=1)
            layer1_pred = torch.argmax(layer1_probs, dim=1)

            # 第二层预测（仅对第一层预测为患病的样本）
            diseased_mask_pred  = (layer1_pred == 1)
            layer2_logits = torch.zeros_like(layer1_logits, device=x.device)

            # 硬路由
            # if diseased_mask_pred.any():
            #     x_diseased = x[diseased_mask_pred]
            #     layer2_logits[diseased_mask_pred] = self.layer2(x_diseased)

            # 真值路由
            x_diseased = x[mask]
            layer2_logits[mask] = self.layer2(x_diseased)


            return layer1_logits, layer2_logits, diseased_mask_pred      
        else:
            raise ValueError(f"Unknown layer: {layer}")

    def predict(self, x):
        with torch.no_grad():
            # 第一层：健康 vs 患病
            layer1_logits = self.layer1(x)
            layer1_probs = torch.softmax(layer1_logits, dim=1)
            layer1_pred = torch.argmax(layer1_probs, dim=1)

            batch_size = x.shape[0]
            final_labels = torch.zeros(batch_size, dtype=torch.long, device=x.device)

            # 健康样本保持为0
            healthy_mask = (layer1_pred == 0)
            final_labels[healthy_mask] = 0

            # 第二层：有合并 vs 无合并（仅对患病样本）
            diseased_mask = (layer1_pred == 1)
            if diseased_mask.any():
                x_diseased = x[diseased_mask]
                layer2_logits = self.layer2(x_diseased)
                layer2_probs = torch.softmax(layer2_logits, dim=1)
                layer2_pred = torch.argmax(layer2_probs, dim=1)
                # 患病样本根据第二层结果分为1或2
                final_labels[diseased_mask] = layer2_pred + 1

            return layer1_logits, layer2_logits,  final_labels

        