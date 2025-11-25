import torch
import logging
from tqdm import tqdm
import os
from utils import plot_training_loss_curves, plot_hierachical_confusion_matrix

logger = logging.getLogger('__main__')

def layer1_train_runner(config, model, trainer, epochs):
    # 第一层调度器
    scheduler_layer1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer_layer1, epochs, eta_min=1e-5
    )

    best_acc = 0.0

    logger.info("=" * 80)
    logger.info("阶段 1: 训练第一层（健康 vs 患病）")
    logger.info("=" * 80)

    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = True

    train_loss = []
    test_loss = []
    for epoch in tqdm(range(epochs), desc='Layer1 Training'):
        metrics_train = trainer.train_layer1_epoch(epoch)
        train_loss.append(metrics_train['loss'])
        trainer.evaluate_layer1(flag='train', epoch_num=epoch)
        # scheduler_layer1.step()

        metrics_test = trainer.evaluate_layer1(flag='test', epoch_num=epoch)
        scheduler_layer1.step()
        test_loss.append(metrics_test['loss'])

        if metrics_test['accuracy'] > best_acc:
            best_acc = metrics_test['accuracy']
            cm = metrics_test['confusion_matrix']
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'model_best.pth'))

    return train_loss, test_loss, cm


def layer2_train_runner(config, model, trainer, epochs):
    # 第二层调度器
    scheduler_layer2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer_layer2, epochs, eta_min=1e-5
    )

    logger.info("=" * 80)
    logger.info("阶段 2: 训练第二层（有合并 vs 无合并半月板损伤）")
    logger.info("=" * 80)

    best_acc = 0.0

    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = True

    if config['training_mode'] == 'both':
        pretrained_weights_path = os.path.join(config['save_dir'], 'model_best.pth')
        pretrained_weights = torch.load(pretrained_weights_path)
        model.load_state_dict(pretrained_weights, strict=False)

    train_loss = []
    test_loss = []
    for epoch in tqdm(range(epochs), desc='Layer2 Training'):
        metrics_train = trainer.train_layer2_epoch(epoch)
        train_loss.append(metrics_train['loss'])
        trainer.evaluate_layer2(flag='train', epoch_num=epoch)
        scheduler_layer2.step()

        if (epoch + 1) % config.get('val_interval', 5) == 0:
            metrics_test = trainer.evaluate_layer2(flag='test', epoch_num=epoch)
            test_loss.append(metrics_test['loss'])

            if metrics_test['accuracy'] > best_acc:
                best_acc = metrics_test['accuracy']
                cm = metrics_test['confusion_matrix']
                torch.save(model.state_dict(), os.path.join(config['save_dir'], 'model_best.pth'))

    return train_loss, test_loss, cm


def joint_train_runner(config, model, trainer, epochs):
    # 联合调度器
    scheduler_joint = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer_joint, epochs, eta_min=1e-5
    )

    logger.info("=" * 80)
    logger.info("阶段 3: 联合训练第一层和第二层")
    logger.info("=" * 80)

    best_acc = 0.0

    for param in model.parameters():
        param.requires_grad = True

    if config['training_mode'] == 'both':
        pretrained_weights_path = os.path.join(config['save_dir'], 'model_best.pth')
        pretrained_weights = torch.load(pretrained_weights_path)
        model.load_state_dict(pretrained_weights, strict=False)

    train_loss = []
    test_loss = []
    for epoch in tqdm(range(epochs), desc='Joint Training'):
        metrics_train = trainer.train_joint_epoch(epoch)
        train_loss.append(metrics_train['loss'])
        trainer.evaluate_hierarchical(flag='train', epoch_num=epoch)
        scheduler_joint.step()

        # 每隔几个 epoch 评估一次
        if (epoch + 1) % config.get('val_interval', 5) == 0:
            metrics_test = trainer.evaluate_hierarchical(flag='test', epoch_num=epoch)
            test_loss.append(metrics_test['loss'])

            if metrics_test['accuracy'] > best_acc:
                best_acc = metrics_test['accuracy']
                cm = metrics_test['confusion_matrix']
                torch.save(model.state_dict(), os.path.join(config['save_dir'], 'model_best.pth'))

    return train_loss, test_loss, cm


def hierarchical_train_runner(config, model, trainer):

    epochs_layer1 = config.get('epochs_layer1', 100)
    epochs_layer2 = config.get('epochs_layer2', 50)
    epochs_joint = config.get('epochs_joint', 20)

    train_loss = []
    test_loss = []
    cm = []
    
    if config['training_mode'] == 'layer1':
        train_loss, test_loss, cm = layer1_train_runner(config, model, trainer, epochs_layer1)
        plot_training_loss_curves(train_loss, test_loss, config['save_dir'])
        plot_hierachical_confusion_matrix(cm=cm, class_names=['健康', '左腿患病', '右腿患病'], save_dir=config['save_dir'])

    elif config['training_mode'] == 'layer2':
        train_loss, test_loss, cm = layer2_train_runner(config, model, trainer, epochs_layer2)
        plot_training_loss_curves(train_loss, test_loss, config['save_dir'])
        plot_hierachical_confusion_matrix(cm=cm, class_names=['有合并半月板损伤', '无合并半月板损伤'], save_dir=config['save_dir'])

    elif config['training_mode'] == 'both':
        layer1_train_runner(config, model, trainer, epochs_layer1)
        layer2_train_runner(config, model, trainer, epochs_layer2)
        train_loss, test_loss, cm = joint_train_runner(config, model, trainer, epochs_joint)
        plot_training_loss_curves(train_loss, test_loss, config['save_dir'])
        plot_hierachical_confusion_matrix(cm=cm, class_names=['健康', '有合并半月板损伤', '无合并半月板损伤'], save_dir=config['save_dir'])

    return train_loss, test_loss, cm