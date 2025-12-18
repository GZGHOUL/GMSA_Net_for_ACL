import logging
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# 自定义模块导入
import Dataset.data_loader as data_loader
from utils import Setup, Environment_Initialization, single_branch_dataset, multi_branch_dataset
from utils import create_balanced_sampler
from Models.hierarchical_net import hierarchical_net
from Models.loss_function import get_loss_function
from Models.optimizers import get_optimizer
from hierarchical_training import HierarchicalTrainer
from hierarchical_visualization import hierarchical_train_runner

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()

# ==============================================================================
# 1. 参数配置 (Arguments Configuration)
# ==============================================================================

# I/O
parser.add_argument('--dataset_path', default='Dataset/Gait3/', help='Data path for gait dataset')
parser.add_argument('--output_dir', default='Results', help='Root output directory')
parser.add_argument('--Norm', type=bool, default=True, help='Enable Data Normalization')

# Dataset Settings
parser.add_argument('--layer1_task', type=str, default='ternary', choices=['binary', 'ternary'],
                    help='Layer 1 task: binary (Healthy vs Sick) or ternary (Healthy vs ACL vs OA)')
parser.add_argument("--use_b_branch", type=bool, default=False, help="Enable B branch (period normalized)")
parser.add_argument("--use_cycle_segmentation", type=bool, default=True, help="Enable A branch switch to period normalized")
parser.add_argument("--label_type", default='injure_label', type=str, help="Label type: injure_label injure_leg_label")
parser.add_argument("--use_layer1_augmentation", type=bool, default=False, help="Enable data augmentation")
parser.add_argument("--use_layer2_augmentation", type=bool, default=False, help="Enable data augmentation")
parser.add_argument("--n_augments", type=int, default=20, help="Number of augmentations per sample (Mirror+Random)")
parser.add_argument("--use_diff_channel", type=bool, default=False, help="Add diff and abs_diff channels to input")

# Model Hyperparameters
parser.add_argument("--model_layer1", default='Layer1_AsyncDualStream_Lite_clinical', type=str, help="model name of layer1")
parser.add_argument("--model_layer2", default='Layer2_AsyncDualStream_Lite_clinical', type=str, help="model name of layer2")
parser.add_argument("--num_heads", default=4, type=int, help="Number of attention heads")
parser.add_argument("--embed_dim", default=64, type=int, help="Embedding dimension")
parser.add_argument("--dim_ff", default=128, type=int, help="Feedforward dimension")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
parser.add_argument("--num_classes", default=3, type=int, help="Number of classes for final output")

# Training Hyperparameters
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate (Base)")
parser.add_argument("--weight_decay", default=1e-1, type=float, help="Weight decay")
parser.add_argument('--training_mode', default='layer1', type=str, choices=['layer1', 'layer2', 'both'], help='Select training phase')
parser.add_argument("--cycle_voting", default=True, type=bool, help="whether use cycle voting")

# Layer-specific Epochs (可以在这里微调各阶段轮数)
parser.add_argument("--epochs_layer1", default=50, type=int)
parser.add_argument("--epochs_layer2", default=50, type=int)
parser.add_argument("--epochs_joint", default=20, type=int)

# System
parser.add_argument('--gpu', type=str, default='0', help='GPU index')
parser.add_argument('--seed', default=2025, type=int, help='Random seed')
parser.add_argument('--console', action='store_true', help="Optimize printout")

def get_arg():
    return parser.parse_args()

def create_dataloader(dataset, batch_size, sampler=None, shuffle=True, num_workers=0):
    """辅助函数: 统一创建DataLoader"""
    if sampler is not None:
        return DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle, 
            pin_memory=False, 
            num_workers=num_workers,
            drop_last=False
        )
    else:
        return DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle, 
            pin_memory=False, 
            num_workers=num_workers,
            drop_last=False
        )

def main():
    # 1. 初始化环境
    config = Setup(get_arg())
    device = Environment_Initialization(config)
    config['device'] = device
    
    # 2. 加载数据
    logger.info(f"正在加载数据... [Mode: {config['training_mode']}]")
    Data = data_loader.load_gait_data(config)

    # 设置序列参数供模型使用
    config['len_ts'] = Data['train_data_a'].shape[2]
    if 'train_data_b' in Data:
        config['len_ts_b'] = Data['train_data_b'].shape[2]
    else:
        config['len_ts_b'] = 100 # 默认值防止报错
    config['ts_dim'] = Data['train_data_a'].shape[1]

    # 3. 构建数据集 (Datasets)
    # 根据是否使用 B 分支选择 Dataset 类
    if config['use_b_branch'] and 'train_data_b' in Data:
        DatasetClass = multi_branch_dataset
        # Joint 数据集需要 B 分支数据
        train_dataset_joint = DatasetClass(Data['train_data_a'], Data['train_data_b'], Data['train_label_a'])
        test_dataset_joint = DatasetClass(Data['test_data_a'], Data['test_data_b'], Data['test_label_a'])
    else:
        DatasetClass = single_branch_dataset
        train_dataset_joint = DatasetClass(Data['train_data_a'], Data['train_label_a'])
        test_dataset_joint = DatasetClass(Data['test_data_a'], Data['test_label_a'])

    # Layer 1 & 2 始终是单分支输入 (根据目前架构)
    train_dataset_layer1 = single_branch_dataset(Data['train_data_layer1'], Data['train_label_layer1'], device=device)
    train_dataset_layer2 = single_branch_dataset(Data['train_data_layer2'], Data['train_label_layer2'], device=device)
    
    test_dataset_layer1 = single_branch_dataset(Data['test_data_layer1'], Data['test_label_layer1'], device=device)
    test_dataset_layer2 = single_branch_dataset(Data['test_data_layer2'], Data['test_label_layer2'], device=device)

    if config['cycle_voting']:
        test_dataset_layer1 = data_loader.CycleVotingDataset(
            Data['test_data_layer1'],  # 传入 (N, 12, 600)
            Data['test_label_layer1'],
            config
        )

    # 4. 构建数据加载器 (DataLoaders)
    # 统一使用 config['batch_size']
    bs = config['batch_size']
    workers = 0 # Windows下建议0，Linux可设为4
    
    # Layer 2: 制作平衡采样器
    train_labels_l2 = Data['train_label_layer2']
    sampler_l2 = create_balanced_sampler(train_labels_l2)

    train_loader_layer1 = create_dataloader(train_dataset_layer1, bs, shuffle=True, num_workers=workers)
    train_loader_layer2 = create_dataloader(train_dataset_layer2, bs, sampler=sampler_l2, shuffle=False, num_workers=workers)
    train_loader_joint  = create_dataloader(train_dataset_joint,  bs, shuffle=True, num_workers=workers)

    test_loader_layer1 = create_dataloader(test_dataset_layer1, bs, shuffle=False, num_workers=workers)
    test_loader_layer2 = create_dataloader(test_dataset_layer2, bs, shuffle=False, num_workers=workers)
    test_loader_joint  = create_dataloader(test_dataset_joint,  bs, shuffle=False, num_workers=workers)

    if config['cycle_voting']:
        test_loader_layer1 = DataLoader(
            dataset=test_dataset_layer1,
            batch_size=1,  # <--- 必须是 1
            shuffle=False,
            num_workers=0,
            collate_fn=None  # 默认即可，因为 BS=1，输出就是 (1, K, C, T)
        )

    # 5. 初始化模型
    logger.info("初始化 Hierarchical ShapeFormer 模型...")
    model = hierarchical_net(config)
    model = model.to(device)

    # 6. 定义损失函数 (Loss Functions)
    # 【关键修改】根据 task 类型决定 Layer 1 的类别数
    num_classes_l1 = 2 if config['layer1_task'] == 'binary' else 3
    loss_function_layer1 = get_loss_function(loss_function='CE', device=device, training_mode='layer1', num_classes=num_classes_l1)
    loss_function_layer2 = get_loss_function(loss_function='FocalLoss', device=device, training_mode='layer2', num_classes=2)

    # 7. 定义优化器 (Optimizers)
    # 【关键修改】移除硬编码，使用 config 中的参数
    optimizer_cls = get_optimizer("AdamW")
    
    # Layer 1: 基础 LR
    optimizer_layer1 = optimizer_cls(
        model.layer1.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Layer 2: 同样的 LR (或者你可以微调)
    optimizer_layer2 = optimizer_cls(
        model.layer2.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Joint: 较小的 LR 微调
    optimizer_joint = optimizer_cls(
        model.parameters(), 
        lr=config['lr'] * 0.1, 
        weight_decay=config['weight_decay']
    )

    # 将对象存入 config 供 Trainer 使用
    config['optimizer_layer1'] = optimizer_layer1
    config['optimizer_layer2'] = optimizer_layer2
    config['optimizer_joint'] = optimizer_joint
    config['loss_module_layer1'] = loss_function_layer1
    config['loss_module_layer2'] = loss_function_layer2

    # 8. 初始化训练器
    trainer = HierarchicalTrainer(
        model=model,
        dataloader_layer1=train_loader_layer1,
        dataloader_layer1_test=test_loader_layer1,
        dataloader_layer2=train_loader_layer2,
        dataloader_layer2_test=test_loader_layer2,
        dataloader_joint=train_loader_joint,
        dataloader_joint_test=test_loader_joint,
        device=device,
        loss_module_layer1=loss_function_layer1,
        loss_module_layer2=loss_function_layer2,
        optimizer_layer1=optimizer_layer1,
        optimizer_layer2=optimizer_layer2,
        optimizer_joint=optimizer_joint,
        lambda_layer2=1.0,
        l2_reg=None,
        save_dir=config['save_dir'],
        config = config
    )

    # 9. 开始训练流程
    logger.info(f"开始训练流程 -> 模式: {config['training_mode']}")
    hierarchical_train_runner(
        config=config,
        model=model,
        trainer=trainer
    )

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', 
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
