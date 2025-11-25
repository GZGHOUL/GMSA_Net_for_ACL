import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)

class GaitAugmentation:
    """步态数据增强类 - 针对12通道时序数据"""
    
    def __init__(self, config=None):
        """
        参数:
            config: 配置字典，可包含：
                - jitter_std: 抖动标准差 (默认0.03)
                - scale_range: 缩放范围 (默认[0.9, 1.1])
                - rotation_range: 旋转角度范围（度） (默认[-5, 5])
                - time_warp_knots: 时间扭曲节点数 (默认4)
                - magnitude_warp_std: 幅度扭曲标准差 (默认0.2)
                - smooth_sigma: 平滑高斯核标准差 (默认1.0)
        """
        self.config = config or {}
        self.jitter_std = self.config.get('jitter_std', 0.03)
        self.scale_range = self.config.get('scale_range', [0.9, 1.1])
        self.rotation_range = self.config.get('rotation_range', [-5, 5])
        self.time_warp_knots = self.config.get('time_warp_knots', 4)
        self.magnitude_warp_std = self.config.get('magnitude_warp_std', 0.2)
        self.smooth_sigma = self.config.get('smooth_sigma', 1.0)
    
    def jitter(self, X):
        """
        添加高斯噪声（抖动）
        模拟测量误差
        X: (n_samples, n_channels, seq_len)
        """
        noise = np.random.normal(0, self.jitter_std, X.shape)
        return X + noise
    
    def scaling(self, X):
        """
        幅度缩放
        模拟不同个体的运动幅度差异
        """
        scale_factors = np.random.uniform(
            self.scale_range[0], 
            self.scale_range[1], 
            (X.shape[0], X.shape[1], 1)
        )
        return X * scale_factors
    
    def rotation(self, X):
        """
        旋转增强（仅针对角度通道：内外翻、内外旋、屈伸角）
        模拟不同的姿态基线
        X: (n_samples, n_channels, seq_len)
        """
        X_aug = X.copy()
        # 角度通道索引：左膝[0,1,2]，右膝[6,7,8]
        angle_channels = [0, 1, 2, 6, 7, 8]
        
        for i, sample in enumerate(X_aug):
            for ch in angle_channels:
                rotation_angle = np.random.uniform(
                    self.rotation_range[0], 
                    self.rotation_range[1]
                )
                X_aug[i, ch] += rotation_angle
        
        return X_aug
    
    def time_warp(self, X):
        """
        时间扭曲
        模拟不同的步态速度变化
        X: (n_samples, n_channels, seq_len)
        """
        n_samples, n_channels, seq_len = X.shape
        X_aug = np.zeros_like(X)
        
        for i in range(n_samples):
            # 随机生成扭曲点
            warp_steps = np.linspace(0, seq_len - 1, self.time_warp_knots)
            random_warps = np.random.randn(self.time_warp_knots) * seq_len * 0.1
            random_warps = np.clip(random_warps, -seq_len * 0.2, seq_len * 0.2)
            
            # 创建平滑的时间映射
            warp_steps_new = warp_steps + random_warps
            warp_steps_new = np.clip(warp_steps_new, 0, seq_len - 1)
            warp_steps_new = np.sort(warp_steps_new)  # 保持单调性
            
            # 插值函数
            time_warp_func = interp1d(
                warp_steps_new, 
                warp_steps, 
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # 应用时间扭曲
            warped_time = time_warp_func(np.arange(seq_len))
            warped_time = np.clip(warped_time, 0, seq_len - 1)
            
            for ch in range(n_channels):
                interp_func = interp1d(
                    np.arange(seq_len),
                    X[i, ch],
                    kind='cubic',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                X_aug[i, ch] = interp_func(warped_time)
        
        return X_aug
    
    def magnitude_warp(self, X):
        """
        幅度扭曲
        模拟不同时刻的运动强度变化
        """
        n_samples, n_channels, seq_len = X.shape
        X_aug = X.copy()
        
        for i in range(n_samples):
            # 为每个通道生成平滑的扭曲曲线
            for ch in range(n_channels):
                warp_curve = np.random.randn(seq_len) * self.magnitude_warp_std
                warp_curve = gaussian_filter1d(warp_curve, sigma=seq_len/10)
                warp_multiplier = 1 + warp_curve
                X_aug[i, ch] *= warp_multiplier
        
        return X_aug
    
    def window_slice(self, X, slice_ratio=0.9):
        """
        窗口切片
        随机截取一段连续的子序列，然后插值回原长度
        模拟步态周期的不同阶段
        """
        n_samples, n_channels, seq_len = X.shape
        X_aug = np.zeros_like(X)
        
        slice_len = int(seq_len * slice_ratio)
        
        for i in range(n_samples):
            # 随机选择起始点
            start_idx = np.random.randint(0, seq_len - slice_len + 1)
            end_idx = start_idx + slice_len
            
            for ch in range(n_channels):
                # 提取子序列
                sliced = X[i, ch, start_idx:end_idx]
                # 插值回原长度
                interp_func = interp1d(
                    np.linspace(0, 1, slice_len),
                    sliced,
                    kind='cubic',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                X_aug[i, ch] = interp_func(np.linspace(0, 1, seq_len))
        
        return X_aug
    
    def left_right_swap(self, X):
        """
        左右腿互换
        针对对称性疾病，这是合理的增强
        X: (n_samples, 12, seq_len) -> 左膝[0:6], 右膝[6:12]
        """
        X_aug = X.copy()
        # 交换左右膝的数据
        X_aug[:, :6, :] = X[:, 6:12, :]  # 左膝 <- 右膝
        X_aug[:, 6:12, :] = X[:, :6, :]  # 右膝 <- 左膝
        return X_aug
    
    def channel_dropout(self, X, dropout_prob=0.1):
        """
        通道dropout
        随机将某些通道置零，模拟数据缺失
        """
        X_aug = X.copy()
        n_samples, n_channels, seq_len = X.shape
        
        for i in range(n_samples):
            for ch in range(n_channels):
                if np.random.rand() < dropout_prob:
                    X_aug[i, ch] = 0
        
        return X_aug
    
    def mixup(self, X, y, alpha=0.2):
        """
        Mixup数据增强
        混合同类别的样本
        """
        n_samples = X.shape[0]
        X_aug = np.zeros_like(X)
        y_aug = np.zeros_like(y)
        
        # 按类别分组
        unique_labels = np.unique(y)
        
        for label in unique_labels:
            label_mask = (y == label)
            label_indices = np.where(label_mask)[0]
            
            if len(label_indices) < 2:
                # 如果该类别样本太少，直接复制
                X_aug[label_mask] = X[label_mask]
                y_aug[label_mask] = y[label_mask]
                continue
            
            for idx in label_indices:
                # 随机选择同类别的另一个样本
                other_idx = np.random.choice(
                    [i for i in label_indices if i != idx]
                )
                
                # 生成混合系数
                lam = np.random.beta(alpha, alpha)
                
                # 混合样本
                X_aug[idx] = lam * X[idx] + (1 - lam) * X[other_idx]
                y_aug[idx] = label  # 保持原标签
        
        return X_aug, y_aug
    
    def augment_batch(self, X, y, n_augments=5, methods=None):
        """
        批量增强
        对每个样本应用多种增强方法
        
        参数:
            X: (n_samples, n_channels, seq_len)
            y: (n_samples,)
            n_augments: 每个样本生成的增强样本数
            methods: 使用的增强方法列表，默认为所有方法
        
        返回:
            X_aug: 增强后的数据
            y_aug: 对应的标签
        """
        if methods is None:
            # 默认增强方法组合
            methods = [
                'jitter',
                'scaling',
                'rotation',
                'time_warp',
                'magnitude_warp',
                'window_slice',
                'left_right_swap'
            ]
        
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(n_augments):
            X_temp = X.copy()
            
            # 随机选择1-3个增强方法组合
            n_methods = np.random.randint(1, min(4, len(methods) + 1))
            selected_methods = np.random.choice(methods, n_methods, replace=False)
            
            for method in selected_methods:
                if method == 'jitter':
                    X_temp = self.jitter(X_temp)
                elif method == 'scaling':
                    X_temp = self.scaling(X_temp)
                elif method == 'rotation':
                    X_temp = self.rotation(X_temp)
                elif method == 'time_warp':
                    X_temp = self.time_warp(X_temp)
                elif method == 'magnitude_warp':
                    X_temp = self.magnitude_warp(X_temp)
                elif method == 'window_slice':
                    X_temp = self.window_slice(X_temp)
                elif method == 'left_right_swap':
                    X_temp = self.left_right_swap(X_temp)
            
            X_augmented.append(X_temp)
            y_augmented.append(y.copy())
        
        return np.vstack(X_augmented), np.hstack(y_augmented)


def augment_layer2_data(X, y, n_augments=5, config=None):
    """
    便捷函数：针对Layer2数据进行增强
    
    参数:
        X: (n_samples, n_channels, seq_len)
        y: (n_samples,)
        n_augments: 每个样本生成的增强样本数
        config: 增强配置
    
    返回:
        X_aug, y_aug
    """
    augmentor = GaitAugmentation(config)
    
    # 对少数类进行更多增强
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    
    X_augmented = []
    y_augmented = []
    
    for label in unique:
        label_mask = (y == label)
        X_label = X[label_mask]
        y_label = y[label_mask]
        
        # 计算需要的增强倍数
        current_count = label_mask.sum()
        augment_ratio = int(np.ceil(max_count / current_count))
        augment_ratio = min(augment_ratio, n_augments)  # 限制最大增强倍数
        
        logger.info(f"类别 {label}: 原始样本 {current_count}, 增强倍数 {augment_ratio}")
        
        # 增强
        X_aug, y_aug = augmentor.augment_batch(X_label, y_label, augment_ratio)
        
        X_augmented.append(X_aug)
        y_augmented.append(y_aug)
    
    return np.vstack(X_augmented), np.hstack(y_augmented)

def augment_layer1_data(data, label):
    new_data = []
    new_labels = []
    for idx, sample in enumerate(data):
        left = sample[:6, :]
        right = sample[6:12, :]

        new_left = right
        new_right = left
        new_diff = new_left - new_right

        new_sample = np.concatenate([new_left, new_right, new_diff], axis=0)
        new_data.append(new_sample)

        if label[idx] == 1:
            new_labels.append(2)
        elif label[idx] == 2:
            new_labels.append(1)
    print(np.array(new_data).shape)
    return new_data, new_labels
        
