import torch
import logging
import os
from tqdm import tqdm
from utils import plot_training_loss_curves, plot_hierachical_confusion_matrix

logger = logging.getLogger('__main__')

def _load_best_weights(config, model):
    """辅助函数：加载当前保存的最佳模型权重"""
    path = os.path.join(config['save_dir'], 'model_best.pth')
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        logger.info(f"已加载最佳权重: {path}")
    else:
        logger.warning("未找到最佳权重文件，继续使用当前权重。")

def _run_epoch_loop(config, model, trainer, epochs, optimizer, 
                    train_func_name, eval_func_name, desc):
    """
    通用的训练循环核心逻辑
    """
    # 初始化调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=1e-6
    )

    best_acc = 0.0
    train_losses = []
    test_losses = []
    best_cm = None
    
    # 获取绑定的方法
    train_func = getattr(trainer, train_func_name)
    eval_func = getattr(trainer, eval_func_name)

    # 进度条
    pbar = tqdm(range(epochs), desc=desc)
    
    for epoch in pbar:
        # 1. 训练
        metrics_train = train_func(epoch)
        train_losses.append(metrics_train['loss'])
        
        # 2. 评估
        metrics_train = eval_func(flag='train', epoch_num=epoch)
        metrics_test = eval_func(flag='test', epoch_num=epoch)
        test_losses.append(metrics_test['loss'])
        
        # 3. 调度器步进
        scheduler.step()

        # 4. 保存最佳模型
        if metrics_test['accuracy'] > best_acc:
            best_acc = metrics_test['accuracy']
            best_cm = metrics_test['confusion_matrix']
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'model_best.pth'))
        
        # 更新进度条显示
        pbar.set_postfix({
            'Tr_Loss': f"{metrics_train['loss']:.4f}",
            'Tr_Acc': f"{metrics_train['accuracy']:.4f}", 
            'Te_Loss': f"{metrics_test['loss']:.4f}",
            'Te_Acc': f"{metrics_test['accuracy']:.4f}"
        })

    # 5. 循环结束后，加载最佳模型供后续使用或最终评估
    logger.info(f"[{desc}] 训练结束。最佳验证集准确率: {best_acc:.4f}")
    _load_best_weights(config, model)
    
    return train_losses, test_losses, best_cm

def layer1_train_runner(config, model, trainer, epochs):
    logger.info("=" * 80)
    logger.info("阶段 1: 训练第一层（主腿健康 vs 患病）")
    logger.info("=" * 80)

    # 冻结 Layer2，解冻 Layer1
    for param in model.layer2.parameters(): param.requires_grad = False
    for param in model.layer1.parameters(): param.requires_grad = True

    return _run_epoch_loop(
        config, model, trainer, epochs,
        optimizer=trainer.optimizers['layer1'],  # 【修复】使用字典访问
        train_func_name='train_layer1_epoch',
        eval_func_name='evaluate_layer1',
        desc='Layer1 Training'
    )

def layer2_train_runner(config, model, trainer, epochs):
    logger.info("=" * 80)
    logger.info("阶段 2: 训练第二层（有合并 vs 无合并）")
    logger.info("=" * 80)

    # 冻结 Layer1，解冻 Layer2
    for param in model.layer1.parameters(): param.requires_grad = False
    for param in model.layer2.parameters(): param.requires_grad = True

    if config['training_mode'] == 'both':
        _load_best_weights(config, model)

    return _run_epoch_loop(
        config, model, trainer, epochs,
        optimizer=trainer.optimizers['layer2'], # 【修复】使用字典访问
        train_func_name='train_layer2_epoch',
        eval_func_name='evaluate_layer2',
        desc='Layer2 Training'
    )

def joint_train_runner(config, model, trainer, epochs):
    logger.info("=" * 80)
    logger.info("阶段 3: 联合训练 (Fine-tuning)")
    logger.info("=" * 80)

    for param in model.parameters(): param.requires_grad = True

    if config['training_mode'] == 'both':
        _load_best_weights(config, model)

    return _run_epoch_loop(
        config, model, trainer, epochs,
        optimizer=trainer.optimizers['joint'], # 【修复】使用字典访问
        train_func_name='train_joint_epoch',
        eval_func_name='evaluate_hierarchical',
        desc='Joint Training'
    )

def hierarchical_train_runner(config, model, trainer):
    epochs_layer1 = config.get('epochs_layer1', 100)
    epochs_layer2 = config.get('epochs_layer2', 50)
    epochs_joint = config.get('epochs_joint', 20)

    train_loss = []
    test_loss = []
    cm = []
    
    mode = config['training_mode']
    save_dir = config['save_dir']
    
    # 获取 Layer 1 的任务类型 (默认 binary)
    l1_task = config.get('layer1_task', 'binary')

    if mode == 'layer1':
        train_loss, test_loss, cm = layer1_train_runner(config, model, trainer, epochs_layer1)
        plot_training_loss_curves(train_loss, test_loss, save_dir)
        
        # [关键修改] 根据任务类型动态设置标签
        if l1_task == 'ternary':
            class_names = ['健康', 'ACL断裂', '关节炎']
        else:
            class_names = ['健康', '患病']
            
        plot_hierachical_confusion_matrix(cm, class_names=class_names, save_dir=save_dir)

    elif mode == 'layer2':
        train_loss, test_loss, cm = layer2_train_runner(config, model, trainer, epochs_layer2)
        plot_training_loss_curves(train_loss, test_loss, save_dir)
        plot_hierachical_confusion_matrix(cm=cm, class_names=['有合并半月板损伤', '无合并半月板损伤'], save_dir=save_dir)

    elif mode == 'both':
        # Phase 1
        l1_tr, l1_te, l1_cm = layer1_train_runner(config, model, trainer, epochs_layer1)
        # 保存 Layer 1 的曲线和混淆矩阵 (便于中间检查)
        plot_training_loss_curves(l1_tr, l1_te, save_dir)
        
        # [关键修改] 同样为 both 模式下的 Layer 1 结果绘图
        if l1_task == 'ternary':
            l1_classes = ['健康', 'ACL断裂', '关节炎']
        else:
            l1_classes = ['健康', '患病']
        plot_hierachical_confusion_matrix(l1_cm, l1_classes, save_dir) 
        
        # Phase 2
        l2_tr, l2_te, l2_cm = layer2_train_runner(config, model, trainer, epochs_layer2)
        plot_training_loss_curves(l2_tr, l2_te, save_dir)
        plot_hierachical_confusion_matrix(cm=l2_cm, class_names=['有合并半月板损伤', '无合并半月板损伤'], save_dir=save_dir) 

        # Phase 3
        train_loss, test_loss, cm = joint_train_runner(config, model, trainer, epochs_joint)
        
        # 最终联合模型的输出通常是 3 类 (健康 / 有合并 / 无合并)
        # 这一步取决于你的 joint 评估逻辑最终输出了什么标签
        plot_training_loss_curves(train_loss, test_loss, save_dir)
        plot_hierachical_confusion_matrix(cm=cm, class_names=['健康', '有合并', '无合并'], save_dir=save_dir)

    return train_loss, test_loss, cm