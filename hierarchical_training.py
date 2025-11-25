import torch
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from Models.loss_function import l2_reg_loss
from Models import analysis

logger = logging.getLogger('__main__')
NEG_METRICS = {'loss'}

class HierarchicalTrainer:
    """
    层级训练器：分两阶段训练
    - 阶段1: 训练第一层（健康 vs 患病）
    - 阶段2: 训练第二层（有合并 vs 无合并半月板损伤）
    """
    def __init__(self, model, dataloader_layer1, dataloader_layer1_test, dataloader_layer2, dataloader_layer2_test, dataloader_joint,
                 dataloader_joint_test, device, loss_module_layer1, loss_module_layer2, optimizer_layer1, optimizer_layer2,
                 optimizer_joint, l2_reg=None, lambda_layer2=1.0, print_interval=10, save_dir=None):
        self.model = model
        self.dataloader_layer1 = dataloader_layer1
        self.dataloader_layer1_test = dataloader_layer1_test
        self.dataloader_layer2 = dataloader_layer2
        self.dataloader_layer2_test = dataloader_layer2_test
        self.dataloader_joint = dataloader_joint
        self.dataloader_joint_test = dataloader_joint_test
        self.device = device
        self.loss_module_layer1 = loss_module_layer1
        self.loss_module_layer2 = loss_module_layer2
        self.optimizer_layer1 = optimizer_layer1
        self.optimizer_layer2 = optimizer_layer2
        self.optimizer_joint = optimizer_joint
        self.l2_reg = l2_reg
        self.lambda_layer2 = lambda_layer2  # 第二层损失权重
        self.print_interval = print_interval
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        self.save_dir = save_dir

    
    def train_layer1_epoch(self, epoch_num):
        """训练第一层：健康 vs 患病"""
        self.model.train()
        epoch_loss = 0
        total_samples = 0
        
        for i, batch in enumerate(self.dataloader_layer1):
            X, targets, IDs = batch
            
            X = X.to(self.device)
            targets = targets.to(self.device)
            
            # 第一层前向传播
            predictions = self.model(X, layer='layer1')
            
            loss = self.loss_module_layer1(predictions, targets)
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)
            
            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model.layer1)
            else:
                total_loss = mean_loss
            
            self.optimizer_layer1.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.layer1.parameters(), max_norm=4.0)
            self.optimizer_layer1.step()
            
            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()
        
        epoch_loss = epoch_loss / total_samples
        return {'epoch': epoch_num, 'loss': epoch_loss}

    def train_layer1_epoch_v2(self, epoch_num):
        """训练第一层：健康/左腿断裂/右腿断裂"""
        self.model.train()
        epoch_loss = 0
        total_samples = 0
        
        for i, batch in enumerate(self.dataloader_layer1):
            X, targets, IDs = batch
            left_targets = []
            right_targets = []
            for label in targets:
                if label == 0:
                    left_targets.append(0)
                    right_targets.append(0)
                elif label == 1:
                    left_targets.append(1)
                    right_targets.append(0)
                elif label == 2:
                    left_targets.append(0)
                    right_targets.append(1)
            left_targets = torch.tensor(left_targets)
            right_targets = torch.tensor(right_targets)
            left_targets = left_targets.to(self.device)
            right_targets = right_targets.to(self.device)
            targets = targets.to(self.device)

            left_X = X[:, :6, :]
            left_X = left_X.to(self.device)
            left_predictions = self.model(left_X, layer='layer1')

            right_X = X[:, 6:, :]
            right_X = right_X.to(self.device)
            right_predictions = self.model(right_X, layer='layer1')
            
            left_loss = self.loss_module_layer1(left_predictions, left_targets)
            right_loss = self.loss_module_layer1(right_predictions, right_targets)
            loss = left_loss + right_loss
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)
            
            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model.layer1)
            else:
                total_loss = mean_loss
            
            self.optimizer_layer1.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.layer1.parameters(), max_norm=4.0)
            self.optimizer_layer1.step()
            
            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()
        
        epoch_loss = epoch_loss / total_samples
        return {'epoch': epoch_num, 'loss': epoch_loss}
    
    def train_layer2_epoch(self, epoch_num):
        """训练第二层：有合并 vs 无合并半月板损伤"""
        self.model.train()
        epoch_loss = 0
        total_samples = 0
        
        for i, batch in enumerate(self.dataloader_layer2):
            X, targets, IDs = batch
            
            X = X.to(self.device)
            targets = targets.to(self.device)
            
            # 第二层前向传播
            predictions = self.model(X, layer='layer2')
            
            loss = self.loss_module_layer2(predictions, targets)
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)
            
            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model.layer2)
            else:
                total_loss = mean_loss
            
            self.optimizer_layer2.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.layer2.parameters(), max_norm=4.0)
            self.optimizer_layer2.step()
            
            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()
        
        epoch_loss = epoch_loss / total_samples
        return {'epoch': epoch_num, 'loss': epoch_loss}

    def train_joint_epoch(self, epoch_num):
        """联合训练第一层和第二层"""
        self.model.train()
        epoch_loss = 0
        epoch_loss_layer1 = 0
        epoch_loss_layer2 = 0
        total_samples = 0
        
        for i, batch in enumerate(self.dataloader_joint):
            X, targets_3class, IDs = batch
            
            X = X.to(self.device)
            targets_3class = targets_3class.to(self.device)
            
            # 构造两层的标签
            # Layer1: 健康(0) vs 患病(1)
            targets_layer1 = (targets_3class > 0).long()  # 0->0, 1->1, 2->1

            # Layer2: 有合并(0) vs 无合并(1) - 仅对患病样本
            diseased_mask = (targets_3class > 0)
            targets_layer2 = torch.zeros_like(targets_3class, device=self.device)
            targets_layer2[diseased_mask] = targets_3class[diseased_mask] - 1  # 1->0, 2->1

            # 联合训练
            layer1_logits, layer2_logits, diseased_mask_pred = self.model(X, mask = diseased_mask, layer='both')
            # targets_layer2[diseased_mask_pred] = targets_3class[diseased_mask_pred] - 1  # 1->0, 2->1
            
            loss_layer1 = self.loss_module_layer1(layer1_logits, targets_layer1)
            loss_layer1_mean = loss_layer1.mean()

            if diseased_mask.any():
                loss_layer2 = self.loss_module_layer2(layer2_logits[diseased_mask], targets_layer2[diseased_mask])
                loss_layer2_mean = loss_layer2.mean()
            else:
                loss_layer2_mean = torch.tensor(0.0, device=self.device)

            # 联合损失
            lambda_layer2 = getattr(self, 'lambda_layer2', 1.0)  # 默认权重为1.0
            total_loss = loss_layer1_mean + lambda_layer2 * loss_layer2_mean

            
            if self.l2_reg:
                total_loss = total_loss + self.l2_reg * l2_reg_loss(self.model)
            
            self.optimizer_joint.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer_joint.step()
            
            with torch.no_grad():
                batch_size = len(targets_3class)
                total_samples += batch_size
                epoch_loss += total_loss.item() * batch_size
                epoch_loss_layer1 += loss_layer1_mean.item() * batch_size
                epoch_loss_layer2 += loss_layer2_mean.item() * batch_size
        
        # 返回详细的损失信息
        return {
            'epoch': epoch_num,
            'loss': epoch_loss / total_samples,
            'loss_layer1': epoch_loss_layer1 / total_samples,
            'loss_layer2': epoch_loss_layer2 / total_samples
        }

    def evaluate_layer1(self, flag='train', epoch_num=None):
        """评估第一层：健康(0) vs 患病(1) 二分类"""
        logger = logging.getLogger('__main__')

        self.model.eval()
        all_predictions = []
        all_targets = []
        epoch_loss = 0
        total_samples = 0

        with torch.no_grad():
            if flag == 'train':
                for batch in self.dataloader_layer1:
                    X, targets, IDs = batch

                    X = X.to(self.device)
                    targets = targets.to(self.device)

                    # 第一层预测
                    layer1_logits = self.model(X, layer='layer1')
                    layer1_pred = torch.argmax(layer1_logits, dim=1)

                    all_predictions.append(layer1_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            if flag == 'test':
                for batch in self.dataloader_layer1_test:
                    X, targets, IDs = batch

                    X = X.to(self.device)
                    targets = targets.to(self.device)

                    # 第一层预测
                    layer1_logits = self.model(X, layer='layer1')
                    loss = self.loss_module_layer2(layer1_logits, targets)
                    batch_loss = torch.sum(loss)

                    total_samples += len(loss)
                    epoch_loss += batch_loss.item()

                    layer1_pred = torch.argmax(layer1_logits, dim=1)

                    all_predictions.append(layer1_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                epoch_loss = epoch_loss / total_samples

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # ✅ 使用 sklearn 一次性计算所有指标
        report = classification_report(
            targets, predictions,
            digits=4,
            output_dict=True,
            zero_division=0
        )

        balanced_acc = balanced_accuracy_score(targets, predictions)
        cm = confusion_matrix(targets, predictions)
        
        # 打印详细报告
        logger.info("="*60)
        logger.info("第一层评估")
        logger.info("="*60)
        logger.info(report)
        logger.info("混淆矩阵:")
        logger.info(cm)
        logger.info("="*60)

        return {
            'epoch': epoch_num,
            'accuracy': report['accuracy'],
            'balanced_accuracy': balanced_acc,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'confusion_matrix': cm,
            'report': report,
            'loss': epoch_loss
        }

    def evaluate_layer1_v2(self, flag='train', epoch_num=None):
        """评估第一层：健康/左腿断裂/右腿断裂 三分类"""
        logger = logging.getLogger('__main__')

        self.model.eval()
        all_predictions = []
        all_targets = []
        epoch_loss = 0
        total_samples = 0

        with torch.no_grad():
            if flag == 'train':
                for batch in self.dataloader_layer1:
                    X, targets, IDs = batch
                    targets = targets.to(self.device)

                    left_X = X[:, :6, :]
                    left_X = left_X.to(self.device)
                    left_logits = self.model(left_X, layer='layer1')
                    left_pred = torch.argmax(left_logits, dim=1)

                    right_X = X[:, 6:, :]
                    right_X = right_X.to(self.device)
                    right_logits = self.model(right_X, layer='layer1')
                    right_pred = torch.argmax(right_logits, dim=1)

                    layer1_pred = []
                    for i in range(X.shape[0]):
                        if left_pred[i] == 0 and right_pred[i] == 0:
                            layer1_pred.append(0)
                        elif left_pred[i] == 1 and right_pred[i] == 0:
                            layer1_pred.append(1)
                        elif left_pred[i] == 0 and right_pred[i] == 1:
                            layer1_pred.append(2)
                        else:
                            left_logits = F.softmax(left_logits, dim=1)
                            right_logits = F.softmax(right_logits, dim=1)
                            if left_logits[i][1] > right_logits[i][1]:
                                layer1_pred.append(1)
                            elif left_logits[i][1] < right_logits[i][1]:
                                layer1_pred.append(2)
                    layer1_pred = torch.tensor(layer1_pred)

                    all_predictions.append(layer1_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            if flag == 'test':
                for batch in self.dataloader_layer1_test:
                    X, targets, IDs = batch
                    left_targets = []
                    right_targets = []
                    for label in targets:
                        if label == 0:
                            left_targets.append(0)
                            right_targets.append(0)
                        elif label == 1:
                            left_targets.append(1)
                            right_targets.append(0)
                        elif label == 2:
                            left_targets.append(0)
                            right_targets.append(1)

                    left_targets = torch.tensor(left_targets)
                    right_targets = torch.tensor(right_targets)

                    left_targets = left_targets.to(self.device)
                    right_targets = right_targets.to(self.device)
                    targets = targets.to(self.device)

                    left_X = X[:, :6, :]
                    left_X = left_X.to(self.device)
                    left_logits = self.model(left_X, layer='layer1')

                    right_X = X[:, 6:, :]
                    right_X = right_X.to(self.device)
                    right_logits = self.model(right_X, layer='layer1')
                    
                    left_loss = self.loss_module_layer1(left_logits , left_targets)
                    right_loss = self.loss_module_layer1(right_logits, right_targets)
                    loss = left_loss + right_loss
                    batch_loss = torch.sum(loss)

                    total_samples += len(loss)
                    epoch_loss += batch_loss.item()

                    left_pred = torch.argmax(left_logits, dim=1)
                    right_pred = torch.argmax(right_logits, dim=1)
                    
                    layer1_pred = []
                    for i in range(X.shape[0]):
                        if left_pred[i] == 0 and right_pred[i] == 0:
                            layer1_pred.append(0)
                        elif left_pred[i] == 1 and right_pred[i] == 0:
                            layer1_pred.append(1)
                        elif left_pred[i] == 0 and right_pred[i] == 1:
                            layer1_pred.append(2)
                        else:
                            left_logits = F.softmax(left_logits, dim=1)
                            right_logits = F.softmax(right_logits, dim=1)
                            if left_logits[i][1] > right_logits[i][1]:
                                layer1_pred.append(1)
                            elif left_logits[i][1] < right_logits[i][1]:
                                layer1_pred.append(2)
                    layer1_pred = torch.tensor(layer1_pred)

                    all_predictions.append(layer1_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                epoch_loss = epoch_loss / total_samples

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # ✅ 使用 sklearn 一次性计算所有指标
        report = classification_report(
            targets, predictions,
            digits=4,
            output_dict=True,
            zero_division=0
        )

        balanced_acc = balanced_accuracy_score(targets, predictions)
        cm = confusion_matrix(targets, predictions)
        
        # 打印详细报告
        logger.info("="*60)
        logger.info("第一层评估")
        logger.info("="*60)
        logger.info(report)
        logger.info("混淆矩阵:")
        logger.info(cm)
        logger.info("="*60)

        return {
            'epoch': epoch_num,
            'accuracy': report['accuracy'],
            'balanced_accuracy': balanced_acc,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'confusion_matrix': cm,
            'report': report,
            'loss': epoch_loss
        }

    def evaluate_layer2(self, flag='train', epoch_num=None):
        """评估第二层：有合并(0) vs 无合并(1) 二分类（仅患病样本）"""
        logger = logging.getLogger('__main__')
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        epoch_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            if flag == 'train':
                for batch in self.dataloader_layer2:
                    X, targets, IDs = batch
                    
                    X = X.to(self.device)
                    targets = targets.to(self.device)

                    # 第二层预测
                    layer2_logits = self.model(X, layer='layer2')
                    layer2_pred = torch.argmax(layer2_logits, dim=1)

                    all_predictions.append(layer2_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            if flag == 'test':
                for batch in self.dataloader_layer2_test:
                    X, targets, IDs = batch
                    
                    X = X.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 第二层预测
                    layer2_logits = self.model(X, layer='layer2')
                    loss = self.loss_module_layer2(layer2_logits, targets)
                    batch_loss = torch.sum(loss)

                    total_samples += len(loss)
                    epoch_loss += batch_loss.item()
        
                    layer2_pred = torch.argmax(layer2_logits, dim=1)

                    all_predictions.append(layer2_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                epoch_loss = epoch_loss / total_samples

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # ✅ 使用 sklearn 一次性计算所有指标
        report = classification_report(
            targets, predictions,
            target_names=['有合并半月板损伤', '无合并半月板损伤'],
            digits=4,
            output_dict=True,
            zero_division=0
        )
        
        balanced_acc = balanced_accuracy_score(targets, predictions)
        cm = confusion_matrix(targets, predictions)
        
        # 打印详细报告（包含每个类别的指标）
        logger.info("="*60)
        logger.info("第二层评估 (有合并 vs 无合并半月板损伤)")
        logger.info("="*60)
        logger.info(f"准确率: {report['accuracy']:.4f} | 平衡准确率: {balanced_acc:.4f}")
        logger.info("\n" + "   " + classification_report(
            targets, predictions,
            target_names=['有合并半月板损伤', '无合并半月板损伤'],
            digits=4,
            zero_division=0
        ))
        logger.info("混淆矩阵:")
        logger.info(f"              预测有合并  预测无合并")
        logger.info(f"实际有合并    {cm[0,0]:8d}    {cm[0,1]:8d}")
        logger.info(f"实际无合并    {cm[1,0]:8d}    {cm[1,1]:8d}")
        logger.info("="*60)
        
        return {
            'epoch': epoch_num,
            'accuracy': report['accuracy'],
            'balanced_accuracy': balanced_acc,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'recall_class0': report['有合并半月板损伤']['recall'],
            'recall_class1': report['无合并半月板损伤']['recall'],
            'confusion_matrix': cm,
            'report': report,
            'loss': epoch_loss
        }

    def evaluate_hierarchical(self, flag='train', epoch_num=None):
        """层级评估：在完整数据集上评估整个层级模型（三分类）"""
        logger = logging.getLogger('__main__')

        self.model.eval()
        all_predictions = []
        all_targets = []
        epoch_loss = 0
        total_samples = 0

        with torch.no_grad():
            if flag == 'train':
                for batch in self.dataloader_joint:
                    X, targets, IDs = batch

                    X = X.to(self.device)
                    targets = targets.to(self.device)

                    # 联合预测
                    layer1_logits, layer2_logits, healthy_mask, diseased_mask, joint_logits = self.model.predict(X)
                    joint_pred = torch.argmax(joint_logits, dim=1)

                    all_predictions.append(joint_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            if flag == 'test':
                for batch in self.dataloader_joint_test:
                    X, targets, IDs = batch

                    X = X.to(self.device)
                    targets = targets.to(self.device)

                    # 构造两层的标签
                    # Layer1: 健康(0) vs 患病(1)
                    targets_layer1 = (targets > 0).long()  # 0->0, 1->1, 2->1

                    # Layer2: 有合并(0) vs 无合并(1) - 仅对患病样本
                    diseased_mask = (targets > 0)
                    targets_layer2 = torch.zeros_like(targets, device=self.device)
                    targets_layer2[diseased_mask] = targets[diseased_mask] - 1  # 1->0, 2->1

                    # 联合预测
                    layer1_logits, layer2_logits, joint_logits = self.model.predict(X)
                    loss = self.loss_module_layer1(layer1_logits, targets_layer1) + self.lambda_layer2 * self.loss_module_layer2(layer2_logits, targets_layer2)
                    batch_loss = torch.sum(loss)

                    total_samples += len(loss)
                    epoch_loss += batch_loss.item()

                    joint_pred = torch.argmax(joint_logits, dim=1)

                    all_predictions.append(joint_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                epoch_loss = epoch_loss / total_samples

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # ✅ 使用 sklearn 一次性计算所有指标
        report = classification_report(
            targets, predictions,
            target_names=['健康', '有合并半月板损伤', '无合并半月板损伤'],
            digits=4,
            output_dict=True,
            zero_division=0
        )
        
        balanced_acc = balanced_accuracy_score(targets, predictions)
        cm = confusion_matrix(targets, predictions, labels=[0, 1, 2])
        
        # 打印详细报告
        logger.info("="*70)
        logger.info("层级模型整体评估 (三分类)")
        logger.info("="*70)
        logger.info(f"准确率: {report['accuracy']:.4f} | 平衡准确率: {balanced_acc:.4f}")
        logger.info("\n" + "   " + classification_report(
            targets, predictions,
            target_names=['健康', '有合并半月板损伤', '无合并半月板损伤'],
            digits=4,
            zero_division=0
        ))
        logger.info("混淆矩阵:")
        logger.info(f"              预测健康  预测有合并  预测无合并")
        logger.info(f"实际健康      {cm[0,0]:6d}    {cm[0,1]:8d}    {cm[0,2]:8d}")
        logger.info(f"实际有合并    {cm[1,0]:6d}    {cm[1,1]:8d}    {cm[1,2]:8d}")
        logger.info(f"实际无合并    {cm[2,0]:6d}    {cm[2,1]:8d}    {cm[2,2]:8d}")
        logger.info("="*70)
        
        return {
            'epoch': epoch_num,
            'accuracy': report['accuracy'],
            'balanced_accuracy': balanced_acc,
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1': report['macro avg']['f1-score'],
            'confusion_matrix': cm,
            'report': report,
            'loss': epoch_loss
        }
    
