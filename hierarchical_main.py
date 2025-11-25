import logging
import os
import argparse
import Dataset.data_loader as data_loader
from utils import Setup, Environment_Initialization, single_branch_dataset, multi_branch_dataset, create_balanced_sampler
from utils import plot_training_loss_curves, plot_hierachical_confusion_matrix
from Shapelet.shapelet_discovery import ShapeletDiscovery
import torch
from Models.hierarchical_shapeformer import hierarchical_shapeformer
from torch.utils.data import DataLoader
from Models.loss_function import get_loss_function
from Models.optimizers import get_optimizer
from hierarchical_training import HierarchicalTrainer
from hierarchical_visualization import hierarchical_train_runner
from torch.utils.data import WeightedRandomSampler
import numpy as np


logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()

# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--dataset_path', default='Dataset/Gait2/', help='Data path for gait dataset')
parser.add_argument('--output_dir', default='Results', help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=True, help='Data Normalization')

# ------------------------------------- Dataset Configuration for Gait Data --------------------------------------------
parser.add_argument("--use_b_branch", type=bool, default=False, help="enable b branch input (period normalized 100 frames)")
parser.add_argument("--num_labels", default=3, type=int, help="number of labels for gait data")
parser.add_argument("--label_type", default='injure_leg_label', type=str, help="label type for gait data")
parser.add_argument("--use_augmentation", type=bool, default=True, help="whether use augmentation for gait data")
parser.add_argument("--n_augments", type=int, default=10, help="number of augmentations for gait data")

# ------------------------------------- Model Configuration for general ------------------------------------------------
parser.add_argument("--num_heads", default=8, type=int, help="number of attention heads")
parser.add_argument("--dim_ff", default=64, type=int, help="dimension of feedforward")
parser.add_argument("--dropout", default=0.5, type=float, help="dropout")
parser.add_argument("--num_classes", default=3, type=int, help="number of classes for gait")

# ------------------------------------- Training Configuration ---------------------------------------------------------
parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=1e-2, type=float, help="weight decay")
parser.add_argument('--val_interval', type=int, default=1, help='Evaluate on validation every N epochs. Must be >= 1')
parser.add_argument('--key_metric', type=str, default='loss', choices={'loss', 'accuracy', 'precision'}, help='Metric used for defining best epoch')
parser.add_argument('--early_stop_patience', type=int, default=20, help='Early stopping patience (epochs without improvement)')
parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='Minimum improvement to reset patience')
parser.add_argument('--training_mode', default='layer1', type=str, help='select training mode:layer1/layer2/both')

# ------------------------------------- System Configuration  ----------------------------------------------------------
parser.add_argument('--gpu', type=str, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console")
parser.add_argument('--seed', default=2025, type=int, help='Seed used for splitting sets')


def get_arg():
    return parser.parse_args()

    
def main():
    config = Setup(parser.parse_args())
    device = Environment_Initialization(config)
    config['device'] = device
    
    # 训练集数据加载
    logger.info("正在加载膝关节步态训练集数据...")
    Data = data_loader.load_gait_data(config)

    config['len_ts'] = Data['train_data_a'].shape[2]
    config['len_ts_b'] = Data['train_data_b'].shape[2]
    config['ts_dim'] = Data['train_data_a'].shape[1]


    model = hierarchical_shapeformer(config)
    model = model.to(device)

    if config['use_b_branch'] and 'train_data_b' in Data:
        train_dataset = multi_branch_dataset(Data['train_data_a'], Data['train_data_b'], Data['train_label_a'])
        test_dataset = multi_branch_dataset(Data['test_data_a'], Data['test_data_b'], Data['test_label_a'])
    else:
        train_dataset_layer1 = single_branch_dataset(Data['train_data_layer1'], Data['train_label_layer1'])
        train_dataset_layer2 = single_branch_dataset(Data['train_data_layer2'], Data['train_label_layer2'])
        train_dataset_joint = single_branch_dataset(Data['train_data_a'], Data['train_label_a'])
        test_dataset_layer1 = single_branch_dataset(Data['test_data_layer1'], Data['test_label_layer1'])
        test_dataset_layer2 = single_branch_dataset(Data['test_data_layer2'], Data['test_label_layer2'])
        test_dataset_joint = single_branch_dataset(Data['test_data_a'], Data['test_label_a'])

    train_loader_layer1 = DataLoader(dataset=train_dataset_layer1, batch_size=32,
                                     shuffle=True, pin_memory=True, num_workers=0)
    train_loader_layer2 = DataLoader(dataset=train_dataset_layer2, batch_size=32,
                                     shuffle=True, pin_memory=True, num_workers=0)
    train_loader_joint = DataLoader(dataset=train_dataset_joint, batch_size=config['batch_size'],
                                    shuffle=True, pin_memory=True, num_workers=0)

    test_loader_joint = DataLoader(dataset=test_dataset_joint, batch_size=config['batch_size'], shuffle=False, pin_memory=True,
                             num_workers=0)
    test_loader_layer1 = DataLoader(dataset=test_dataset_layer1, batch_size=32, shuffle=False, pin_memory=True,
                                    num_workers=0)
    test_loader_layer2 = DataLoader(dataset=test_dataset_layer2, batch_size=32, shuffle=False, pin_memory=True,
                                    num_workers=0)

    loss_function_layer1 = get_loss_function(loss_function='CE', device=device)
    loss_function_layer2 = get_loss_function(loss_function='CE', device=device)

    optimizer = get_optimizer("AdamW")
    optimizer_layer1 = optimizer(model.layer1.parameters(), lr=1e-4, weight_decay=5e-2)
    optimizer_layer2 = optimizer(model.layer2.parameters(), lr=1e-4, weight_decay=1e-4, warmup=5)
    optimizer_joint = optimizer(model.parameters(), lr=config['lr'] * 0.1, weight_decay=config['weight_decay'])

    config['optimizer_layer1'] = optimizer_layer1
    config['optimizer_layer2'] = optimizer_layer2
    config['optimizer_joint'] = optimizer_joint
    config['loss_module_layer1'] = loss_function_layer1
    config['loss_module_layer2'] = loss_function_layer2

    logger.info("开始模型训练...")
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
        save_dir=config['save_dir']
    )

    _, _, _ = hierarchical_train_runner(
        config=config,
        model=model,
        trainer=trainer
    )



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    main()
