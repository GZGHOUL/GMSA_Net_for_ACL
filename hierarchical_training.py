import torch
import numpy as np
import logging
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from Models.loss_function import l2_reg_loss

logger = logging.getLogger('__main__')

class HierarchicalTrainer:
    """
    层级训练器：分两阶段训练
    - 阶段1 (Layer1): 训练主腿健康 vs 患病 (二分类)
    - 阶段2 (Layer2): 训练患病样本的有合并 vs 无合并 (二分类)
    """
    def __init__(self, model, dataloader_layer1, dataloader_layer1_test, dataloader_layer2, dataloader_layer2_test, dataloader_joint,
                 dataloader_joint_test, device, loss_module_layer1, loss_module_layer2, optimizer_layer1, optimizer_layer2,
                 optimizer_joint, l2_reg=None, lambda_layer2=1.0, print_interval=10, save_dir=None, config=None):
        self.model = model
        # DataLoaders
        self.loaders = {
            'layer1_train': dataloader_layer1,
            'layer1_test': dataloader_layer1_test,
            'layer2_train': dataloader_layer2,
            'layer2_test': dataloader_layer2_test,
            'joint_train': dataloader_joint,
            'joint_test': dataloader_joint_test
        }
        self.device = device
        self.loss_funcs = {
            'layer1': loss_module_layer1,
            'layer2': loss_module_layer2
        }
        self.optimizers = {
            'layer1': optimizer_layer1,
            'layer2': optimizer_layer2,
            'joint': optimizer_joint
        }
        self.l2_reg = l2_reg
        self.lambda_layer2 = lambda_layer2
        self.save_dir = save_dir
        self.config = config
    def _train_epoch_generic(self, layer_name, epoch_num):
        """通用的单层训练循环"""
        self.model.train()
        optimizer = self.optimizers[layer_name]
        loss_func = self.loss_funcs[layer_name]
        loader = self.loaders[f'{layer_name}_train']
        
        epoch_loss = 0
        total_samples = 0
        
        # 确定要优化的参数模块
        model_layer = getattr(self.model, layer_name)

        for batch in loader:
            X, targets, _ = batch
            X, targets = X.to(self.device), targets.to(self.device)
            
            # 前向传播
            predictions = self.model(X, layer=layer_name)
            loss = loss_func(predictions, targets)
            
            # 处理 reduction='none' 的情况
            if loss.ndim > 0:
                batch_loss = torch.sum(loss)
                mean_loss = batch_loss / len(loss)
            else:
                batch_loss = loss.item() * len(targets)
                mean_loss = loss

            # L2 正则
            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(model_layer)
            else:
                total_loss = mean_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_layer.parameters(), max_norm=4.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss
            total_samples += len(targets)
        
        return {'epoch': epoch_num, 'loss': epoch_loss / total_samples}

    def train_layer1_epoch(self, epoch_num):
        return self._train_epoch_generic('layer1', epoch_num)

    def train_layer2_epoch(self, epoch_num):
        return self._train_epoch_generic('layer2', epoch_num)

    def train_joint_epoch(self, epoch_num):
        """联合训练逻辑较复杂，保持独立"""
        self.model.train()
        stats = {'loss': 0, 'loss_l1': 0, 'loss_l2': 0, 'total': 0}
        
        for batch in self.loaders['joint_train']:
            X, targets_3class, _ = batch
            X, targets_3class = X.to(self.device), targets_3class.to(self.device)
            
            # 构造标签
            targets_l1 = (targets_3class > 0).long() # 0->0, 1/2->1
            
            diseased_mask = (targets_3class > 0)
            targets_l2 = torch.zeros_like(targets_3class)
            # 仅在患病样本上计算 L2 Loss (1->0, 2->1)
            targets_l2[diseased_mask] = targets_3class[diseased_mask] - 1
            
            # 联合前向
            l1_logits, l2_logits, _ = self.model(X, mask=diseased_mask, layer='both')
            
            loss_l1 = self.loss_funcs['layer1'](l1_logits, targets_l1).mean()
            
            if diseased_mask.any():
                loss_l2 = self.loss_funcs['layer2'](l2_logits[diseased_mask], targets_l2[diseased_mask]).mean()
            else:
                loss_l2 = torch.tensor(0.0, device=self.device)
                
            total_loss = loss_l1 + self.lambda_layer2 * loss_l2
            
            if self.l2_reg:
                total_loss += self.l2_reg * l2_reg_loss(self.model)
                
            self.optimizers['joint'].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizers['joint'].step()
            
            bs = len(targets_3class)
            stats['total'] += bs
            stats['loss'] += total_loss.item() * bs
            stats['loss_l1'] += loss_l1.item() * bs
            stats['loss_l2'] += loss_l2.item() * bs
            
        return {k: v / stats['total'] for k, v in stats.items() if k != 'total'}

    def evaluate_layer1(self, flag='train', epoch_num=None):
        self.model.eval()
        task_type = self.config.get('layer1_task', 'binary')
        
        # === 训练集评估 (直接计算 Loss/Acc) ===
        if flag == 'train':
            return self._evaluate_standard('layer1', flag, epoch_num)

        # === 测试集评估 ===
        elif flag == 'test':
            if self.config['cycle_voting']:
                loss_func = self.loss_funcs['layer1']
                all_preds, all_targets = [], []
                epoch_loss = 0
                total_samples = 0

                loader = self.loaders['layer1_test']  # 这是 batch_size=1 的 loader

                with torch.no_grad():
                    for batch in loader:
                        # X shape: [1, K, C, T] -> K个周期
                        # targets shape: [1] -> 1个标签
                        X, targets, _ = batch

                        # 移除 batch 维度 -> [K, C, T]
                        X = X.squeeze(0).to(self.device)
                        targets = targets.to(self.device)

                        # 如果 K=0 (该样本没提取出有效周期)，跳过或判错
                        if X.shape[0] == 0:
                            continue

                        # 1. 对 K 个周期分别预测
                        # logits shape: [K, NumClasses]
                        logits = self.model(X, layer='layer1')

                        # 2. 计算 Loss (可选，取平均 logits 的 loss)
                        # 简单的做法是把 target 扩展成 [K]，算所有周期的 loss 平均
                        targets_expanded = targets.repeat(X.shape[0])
                        loss = loss_func(logits, targets_expanded)
                        if loss.ndim > 0:
                            loss = torch.mean(loss)
                        epoch_loss += loss.item()

                        # 3. 投票机制 (Voting Strategy)
                        # 策略 A: 平均概率 (Soft Voting) - 推荐
                        probs = F.softmax(logits, dim=1)  # [K, NumClasses]
                        avg_prob = torch.mean(probs, dim=0)  # [NumClasses]
                        final_pred = torch.argmax(avg_prob).item()

                        # 策略 B: 多数表决 (Hard Voting)
                        # votes = torch.argmax(logits, dim=1)
                        # final_pred = torch.mode(votes).values.item()

                        all_preds.append(final_pred)
                        all_targets.append(targets.item())
                        total_samples += 1

                # 计算指标
                avg_loss = epoch_loss / total_samples if total_samples > 0 else 0

                return self._compute_metrics(
                    all_targets, all_preds, avg_loss, epoch_num,
                    task_name="Layer1 Test (Cycle Voting)",
                    target_names=['健康', 'ACL断裂', '关节炎']
                )
            else:
                # --- 分支 A: 二分类任务 (Binary) ---
                if task_type == 'binary':
                    return self._evaluate_standard('layer1', flag, epoch_num)
                # --- 分支 B: 三分类任务 (Ternary) ---
                else:
                    all_probs = []
                    loader = self.loaders['layer1_test']
                    epoch_loss = 0
                    total_samples = 0
                    all_targets = []

                    with torch.no_grad():
                        for X, targets, _ in loader:
                            X, targets = X.to(self.device), targets.to(self.device)
                            logits = self.model(X, layer='layer1')

                            loss = self.loss_funcs['layer1'](logits, targets)
                            batch_loss = torch.sum(loss).item() if loss.ndim > 0 else loss.item() * len(targets)
                            epoch_loss += batch_loss
                            total_samples += len(targets)

                            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
                            all_targets.append(targets.cpu().numpy())

                    epoch_loss /= total_samples

                    # [N*2, 3]
                    probs_all = np.concatenate(all_probs, axis=0)
                    targets_all = np.concatenate(all_targets, axis=0)

                    # 切分视角
                    num_patients = len(probs_all) // 2
                    probs_orig = probs_all[:num_patients]
                    probs_mirror = probs_all[num_patients:]

                    targets_orig = targets_all[:num_patients]
                    # targets_mirror = targets_all[num_patients:] # 应该与 orig 相同

                    # 简单平均融合 (TTA)
                    final_probs = (probs_orig + probs_mirror) / 2.0
                    final_preds = np.argmax(final_probs, axis=1)

                    # 计算指标
                    return self._compute_metrics(
                        targets_orig, final_preds, epoch_loss, epoch_num,
                        task_name="Layer1 Test (Disease Type)",
                        target_names=['健康', 'ACL断裂', '关节炎']
                    )

    def evaluate_layer2(self, flag='train', epoch_num=None):
        """Layer 2 评估: 二分类 (有合并 vs 无合并)"""
        return self._evaluate_standard(
            'layer2', flag, epoch_num, 
            target_names=['有合并半月板损伤', '无合并半月板损伤']
        )

    def evaluate_hierarchical(self, flag='train', epoch_num=None):
        """联合模型评估"""
        self.model.eval()
        loader = self.loaders[f'joint_{flag}']
        
        all_preds, all_targets = [], []
        epoch_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for X, targets, _ in loader:
                X, targets = X.to(self.device), targets.to(self.device)
                
                # 联合预测逻辑
                l1_logits, l2_logits, _, _, joint_logits = self.model.predict(X)
                
                # 简化的 Loss 计算 (略去复杂的 mask 逻辑，仅作评估参考)
                # 实际应复用 train_joint 的 loss 逻辑
                preds = torch.argmax(joint_logits, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        # 注意：这里 targets 可能包含 5 类，需要映射
        targets = np.concatenate(all_targets)
        if targets.max() > 2:
             mapper = np.array([0, 1, 2, 1, 2])
             targets = mapper[targets]
             
        return self._compute_metrics(
            targets, np.concatenate(all_preds), 0.0, epoch_num,
            task_name="Joint Evaluation",
            target_names=['健康', '有合并', '无合并']
        )

    def _evaluate_standard(self, layer_name, flag, epoch_num, target_names=None):
        """通用的标准评估流程"""
        self.model.eval()
        loader = self.loaders[f'{layer_name}_{flag}']
        loss_func = self.loss_funcs[layer_name]
        
        all_preds, all_targets = [], []
        epoch_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for X, targets, _ in loader:
                X, targets = X.to(self.device), targets.to(self.device)
                logits = self.model(X, layer=layer_name)
                
                loss = loss_func(logits, targets)
                batch_loss = torch.sum(loss).item() if loss.ndim > 0 else loss.item() * len(targets)
                epoch_loss += batch_loss
                total_samples += len(targets)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        metrics = self._compute_metrics(
            np.concatenate(all_targets), np.concatenate(all_preds), 
            epoch_loss / total_samples, epoch_num,
            task_name=f"{layer_name} {flag}", target_names=target_names
        )
        
        return metrics

    def _compute_metrics(self, y_true, y_pred, loss, epoch_num, task_name="Task", target_names=None):
        """统一指标计算与日志打印"""

        # [关键修复] 如果提供了 target_names，必须同时指定 labels 索引列表
        # 这样即使某些类别在当前 batch/epoch 中未出现（例如预测全为0），也不会报错
        labels = None
        if target_names is not None:
            labels = list(range(len(target_names)))

        report = classification_report(
            y_true, y_pred,
            labels=labels,  # <--- 新增这行
            target_names=target_names,
            digits=4,
            output_dict=True,
            zero_division=0
        )

        # 计算混淆矩阵时也建议加上 labels，确保矩阵大小固定 (3x3)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        acc = report['accuracy']

        logger.info("=" * 60)
        logger.info(f"{task_name} Report | Loss: {loss:.4f} | Acc: {acc:.4f}")
        logger.info("-" * 30)

        # 打印简报
        if target_names:
            for name in target_names:
                if name in report:
                    rec = report[name]['recall']
                    pre = report[name]['precision']
                    logger.info(f"  {name:<10}: Recall={rec:.4f}, Prec={pre:.4f}")

        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info("=" * 60)

        metrics = {
            'epoch': epoch_num,
            'accuracy': acc,
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'confusion_matrix': cm,
            'loss': loss,
            'report': report
        }

        # 展平 report 中的 weighted avg 指标方便外部调用
        if 'weighted avg' in report:
            metrics.update({
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score']
            })

        return metrics
    
