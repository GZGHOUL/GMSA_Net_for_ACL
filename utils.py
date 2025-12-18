import os
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, WeightedRandomSampler
from matplotlib import font_manager

# 配置日志格式
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def Setup(args):
    """
    配置输出路径并保存配置信息
    """
    config = args.__dict__
    
    # 使用 Path 对象简化路径操作
    initial_timestamp = datetime.now()
    base_dir = Path(config['output_dir'])
    dataset_name = Path(config.get('dataset_path', 'default_dataset')).name # 获取最后一部分作为名字
    
    # 构造带时间戳的输出目录
    run_dir = base_dir / dataset_name / initial_timestamp.strftime("%Y-%m-%d_%H-%M")
    
    # 更新 config 中的路径
    config['output_dir'] = str(run_dir)
    config['save_dir'] = str(run_dir / 'checkpoints')
    config['pred_dir'] = str(run_dir / 'predictions')
    config['tensorboard_dir'] = str(run_dir / 'tb_summaries')
    
    # 创建目录
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # 保存配置文件
    json_path = run_dir / 'configuration.json'
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(config, fp, indent=4, sort_keys=True, ensure_ascii=False)

    logger.info(f"Stored configuration file in '{run_dir}'")

    return config

def create_dirs(dirs):
    """
    创建目录列表
    """
    try:
        for dir_path in dirs:
            # exist_ok=True 自动处理目录已存在的情况，无需手动 check
            os.makedirs(dir_path, exist_ok=True)
        return 0
    except Exception as err:
        logger.error(f"Creating directories error: {err}")
        exit(-1)

def Environment_Initialization(config):
    """
    初始化计算设备和随机种子
    """
    # 优先使用配置指定的 GPU
    gpu_idx = config.get('gpu', '0')
    if torch.cuda.is_available() and gpu_idx != '-1':
        device = torch.device(f'cuda:{gpu_idx}')
        torch.cuda.set_device(device)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)} (Index: {gpu_idx})")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # 设置随机种子
    seed = config.get('seed', None)
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 为了完全复现，可能需要牺牲一点性能
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")
        
    return device

# ==============================================================================
# 优化后的 Dataset 类 (性能提升关键点)
# ==============================================================================

class single_branch_dataset(Dataset):
    def __init__(self, data, label, device=None):
        super().__init__()
        # 直接转 Tensor
        self.feature = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(label).long()
        
        # 如果传入了 device，直接送入显存！
        if device is not None:
            self.feature = self.feature.to(device)
            self.labels = self.labels.to(device)

    def __getitem__(self, index):
        return self.feature[index], self.labels[index], index

    def __len__(self):
        return len(self.labels)

class multi_branch_dataset(Dataset):
    def __init__(self, data_a, data_b, label):
        super().__init__()
        # 【优化】同上，预先转换为 Tensor
        self.feature_a = torch.from_numpy(data_a).float()
        self.feature_b = torch.from_numpy(data_b).float()
        self.labels = torch.from_numpy(label).long()

    def __getitem__(self, index):
        # 返回元组 (data_a, data_b) 适配模型输入
        return (self.feature_a[index], self.feature_b[index]), self.labels[index], index

    def __len__(self):
        return len(self.labels)

# ==============================================================================
# 采样与绘图工具
# ==============================================================================

def create_balanced_sampler(labels):
    """创建加权随机采样器以解决类别不平衡"""
    # 确保 labels 是 numpy 数组
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    
    # 处理某个类别可能为0的情况，防止除以0
    class_weights = np.zeros_like(class_counts, dtype=np.float32)
    mask = class_counts > 0
    class_weights[mask] = 1.0 / class_counts[mask]
    
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def _set_plot_style():
    """内部辅助函数：设置绘图字体，自动寻找可用中文字体"""
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    
    # 常见的系统自带中文字体列表
    font_candidates = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
    
    # 检查系统已安装字体
    system_fonts = {f.name for f in font_manager.fontManager.ttflist}
    
    for font in font_candidates:
        if font in system_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            return
            
    # 如果找不到中文字体，回退到默认，并打印警告
    logger.warning("未找到常见中文字体 (SimHei等)，绘图中的中文可能显示为方框。")

def plot_training_loss_curves(train_loss, test_loss, save_dir):
    """绘制 Loss 曲线"""
    _set_plot_style()
    
    epochs = range(1, len(train_loss) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, test_loss, 'r-', label='Test Loss', linewidth=2)
    
    ax.set_title('Training Process Monitor - Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Training curves saved to: {save_path}")
    # plt.show() # 在服务器运行时通常不需要 show，或者由主程序控制
    plt.close(fig) # 关闭图形释放内存
    
    return str(save_path)

def plot_hierachical_confusion_matrix(cm, class_names, save_dir):
    """绘制混淆矩阵"""
    _set_plot_style()

    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Sample Count'})

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # 添加准确率标注
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    plt.figtext(0.02, 0.02, f'Overall Acc: {accuracy:.3f}', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    plt.tight_layout()

    save_path = Path(save_dir) / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Confusion matrix saved to: {save_path}")
    plt.close()

    return str(save_path)