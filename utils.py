import os
import json
from datetime import datetime
import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import WeightedRandomSampler

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def Setup(args):
    """输入: args: arguments object from argparse; 返回: config: configuration dictionary"""
    config = args.__dict__ 
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, config['dataset_path'], initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}' as a configuration.json".format(output_dir))

    return config

def create_dirs(dirs):
    """输入: dirs: a list of directories to create, in case these directories are not found; 返回: exit_code: 0 if success, -1 if failure"""
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def Environment_Initialization(config):
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))
        if config['seed'] is not None:
            torch.cuda.manual_seed(config['seed'])
    elif device == 'cpu':
        if config['seed'] is not None:
            torch.manual_seed(config['seed'])
    return device

class single_branch_dataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.feature = data
        self.labels = label.astype(np.int32)

    def __getitem__(self, index):
        x = self.feature[index]
        x = x.astype(np.float32)

        y = self.labels[index]

        data = torch.tensor(x)
        label = torch.tensor(y)

        return data, label, index

    def __len__(self):
        return len(self.labels)

class multi_branch_dataset(Dataset):
    def __init__(self, data_a, data_b, label):
        super().__init__()
        self.feature_a = data_a
        self.feature_b = data_b
        self.labels = label.astype(np.int32)

    def __getitem__(self, index):
        x_a = self.feature_a[index].astype(np.float32)
        x_b = self.feature_b[index].astype(np.float32)
        y = self.labels[index]
        data_a = torch.tensor(x_a)
        data_b = torch.tensor(x_b)
        label = torch.tensor(y)
        return (data_a, data_b), label, index

    def __len__(self):
        return len(self.labels)

# 计算每个样本的采样权重
def create_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # 允许重复采样
    )
    return sampler


def plot_training_loss_curves(train_loss, test_loss, save_dir):
    """绘制训练过程中的loss变化曲线"""
    epochs = range(1, len(train_loss) + 1)
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(1, 1, figsize=(15, 10))
    fig.suptitle('hierarchical训练过程监控', fontsize=16, fontweight='bold')
    
    # Loss曲线
    axes.plot(epochs, train_loss, 'b-', label='训练集', linewidth=2)
    axes.plot(epochs, test_loss, 'r-', label='测试集', linewidth=2)
    axes.set_title('Loss变化曲线', fontsize=14)
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    curve_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    logging.info(f"训练曲线已保存到: {curve_path}")
    plt.show()
    
    return curve_path


def plot_hierachical_confusion_matrix(cm, class_names, save_dir):
    """绘制混淆矩阵"""

    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数量'})

    plt.title('混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)

    # 添加准确率信息
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.02, 0.02, f'总体准确率: {accuracy:.3f}', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    plt.tight_layout()

    # 保存图片
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    logging.info(f"混淆矩阵已保存到: {cm_path}")
    plt.show()

    return cm_path