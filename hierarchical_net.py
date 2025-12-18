import torch
import torch.nn as nn
import logging

# 导入所有可能的子模型类，以便动态调用
from Models.layer1 import Layer1_shapeformer, Layer1_AsyncDualStream_Lite, Layer1_AsyncDualStream_Lite_clinical
from Models.layer2 import Layer2_AsyncDualStream_Lite, Layer2_shapeformer, Layer2_UltraLite, Layer2_AsyncDualStream_Lite_clinical

logger = logging.getLogger(__name__)

class hierarchical_net(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 1. 动态配置子模型
        # 默认使用 AsyncDualStream_Lite，但可以通过 config 修改
        l1_name = config.get('model_layer1', 'Layer1_AsyncDualStream_Lite')
        l2_name = config.get('model_layer2', 'Layer2_AsyncDualStream_Lite')
        
        self.layer1 = self._build_layer(l1_name, config)
        self.layer2 = self._build_layer(l2_name, config)
        
        logger.info(f"Initialized Hierarchical Model: L1={l1_name}, L2={l2_name}")

    def _build_layer(self, model_name, config):
        """工厂方法：根据名称构建模型"""
        models_map = {
            'Layer1_shapeformer': Layer1_shapeformer,
            'Layer1_AsyncDualStream_Lite': Layer1_AsyncDualStream_Lite,
            'Layer1_AsyncDualStream_Lite_clinical': Layer1_AsyncDualStream_Lite_clinical,
            'Layer2_AsyncDualStream_Lite': Layer2_AsyncDualStream_Lite,
            'Layer2_AsyncDualStream_Lite_clinical': Layer2_AsyncDualStream_Lite_clinical,
            'Layer2_shapeformer': Layer2_shapeformer,
            'Layer2_UltraLite': Layer2_UltraLite
        }
        
        if model_name not in models_map:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(models_map.keys())}")
            
        return models_map[model_name](config)
    
    def forward(self, x, mask=None, layer='both'):
        """
        Args:
            x: Input tensor [B, C, T]
            mask: Ground Truth mask for diseased samples (used in 'both' training mode)
            layer: 'layer1', 'layer2', or 'both'
        """
        if layer == 'layer1':
            return self._forward_layer(self.layer1, x)
            
        elif layer == 'layer2':
            return self._forward_layer(self.layer2, x)
            
        elif layer == 'both':
            if mask is None: raise ValueError("Mask required")

            # 1. Layer 1 Forward
            l1_out = self.layer1(x)
            l1_logits = l1_out[0] if isinstance(l1_out, tuple) else l1_out

            # 2. 预测 Mask (逻辑更新)
            l1_pred = torch.argmax(l1_logits, dim=1)
            
            # 【关键修改】判断哪些样本需要送入 Layer 2
            # Binary模式: 1 是患病
            # Ternary模式: 1(左) 和 2(右) 都是患病
            diseased_mask_pred = (l1_pred > 0) 

            # 3. Layer 2 Forward (Teacher Forcing with GT mask)
            l2_logits = torch.zeros(x.shape[0], 2, device=x.device)

            if mask.any():
                x_diseased = x[mask]
                l2_out = self.layer2(x_diseased)
                l2_batch_logits = l2_out[0] if isinstance(l2_out, tuple) else l2_out
                l2_logits[mask] = l2_batch_logits

            return l1_logits, l2_logits, diseased_mask_pred      
        else:
            raise ValueError(f"Unknown layer mode: {layer}")

    def _forward_layer(self, layer_module, x):
        """辅助函数：处理可能返回 tuple 的层"""
        out = layer_module(x)
        if isinstance(out, tuple):
            return out[0] # 只返回 logits
        return out

    def predict(self, x):
        """End-to-end Inference"""
        self.eval()
        with torch.no_grad():
            # 1. Layer 1
            l1_out = self.layer1(x)
            l1_logits = l1_out[0] if isinstance(l1_out, tuple) else l1_out
            l1_pred = torch.argmax(l1_logits, dim=1)

            batch_size = x.shape[0]
            # 初始化 Final Labels (默认为 Layer 1 的预测结果)
            # Binary: 0, 1 -> 映射后变成 0 (健康), ? (待定)
            # Ternary: 0, 1, 2 -> 映射后变成 0, ?, ?
            final_labels = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            l2_logits = torch.zeros(batch_size, 2, device=x.device)

            # 2. 确定患病样本 (Binary: 1, Ternary: 1 or 2)
            diseased_mask = (l1_pred > 0)
            
            # 健康样本直接定为 0
            final_labels[~diseased_mask] = 0 

            if diseased_mask.any():
                x_diseased = x[diseased_mask]
                l2_out = self.layer2(x_diseased)
                l2_batch_logits = l2_out[0] if isinstance(l2_out, tuple) else l2_out
                l2_logits[diseased_mask] = l2_batch_logits
                l2_pred = torch.argmax(l2_batch_logits, dim=1)
                
                final_labels[diseased_mask] = l2_pred + 1

            return l1_logits, l2_logits, final_labels

        