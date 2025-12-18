import os
import numpy as np
import pandas as pd
import logging
import json
import re
from torch.utils.data import Dataset
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from Dataset.gait_augmentation import augment_layer2_data
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# ==========================================
# 1. 基础工具函数 (Utils)
# ==========================================

def parse_features_json(s: str) -> np.ndarray:
    """解析JSON字符串为numpy数组，处理NaN"""
    s = re.sub(r'\bnan\b', 'null', str(s), flags=re.IGNORECASE)
    try:
        data = json.loads(s)
        arr = np.array(data, dtype=float)
    except Exception:
        arr = np.array(eval(s, {"__builtins__": None}, {"nan": float('nan')}), dtype=float)
    return arr

def sanitize_features_array(arr: np.ndarray, clip_value: float = 1e6) -> np.ndarray:
    """数值清洗：处理NaN和Inf"""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=float)
    
    # NaN -> 0, Inf -> limit
    arr = np.nan_to_num(arr, nan=0.0, posinf=clip_value, neginf=-clip_value)
    np.clip(arr, -clip_value, clip_value, out=arr)
    return arr

# ==========================================
# 2. 步态周期处理类 (来自 data_period_norm.py)
# ==========================================

class GaitPreprocessor:
    def __init__(self, target_len=100, flex_ch_idx=2):
        """
        target_len: 归一化后的步态周期长度 (推荐 100)
        flex_ch_idx: 屈伸角所在的通道索引 (根据数据，假设是第3个通道，即索引2)
        """
        self.target_len = target_len
        self.flex_idx = flex_ch_idx

    def process_subject(self, raw_data_12x600):
        """
        输入: 单个受试者的原始数据 (12, 600)
        输出: 该受试者提取出的所有样本 (N_cycles, 18, 100)
        """
        # 1. 拆分左右腿
        left_data = raw_data_12x600[:6, :]   # (6, 600)
        right_data = raw_data_12x600[6:, :]  # (6, 600)

        # 2. 提取并归一化周期
        left_cycles = self._extract_and_normalize(left_data)
        right_cycles = self._extract_and_normalize(right_data)

        # 3. 配对与构建差值流
        # 取最小周期数进行配对 (确保左右腿数据量对齐)
        num_pairs = min(len(left_cycles), len(right_cycles))
        
        if num_pairs == 0:
            return None # 如果没提取到完整周期，丢弃该样本

        processed_samples = []
        for i in range(num_pairs):
            L = left_cycles[i]  # (6, 100)
            R = right_cycles[i] # (6, 100)
                       
            # 堆叠: 通道维变为 6+6=12
            combined = np.concatenate([L, R], axis=0) # (18, 100)
            processed_samples.append(combined)

        return np.array(processed_samples)

    def _extract_and_normalize(self, leg_data):
        """内部函数：单腿周期提取"""
        flexion = leg_data[self.flex_idx, :]
        
        # 寻找波峰 (Peak Detection)
        # distance: 设定波峰最小间距 (600帧里大概4-6个周期，间距设为60比较安全)
        peaks, _ = find_peaks(flexion, distance=60, prominence=0.5)
        
        cycles = []
        # 至少要有2个波峰才能构成一个周期
        if len(peaks) < 2:
            return cycles

        for i in range(len(peaks) - 1):
            start, end = peaks[i], peaks[i+1]
            cycle_raw = leg_data[:, start:end] # (6, current_len)
            
            # 线性插值归一化 (比FFT重采样更稳定，不会产生边缘振铃效应)
            cycle_norm = self._interpolate(cycle_raw)
            cycles.append(cycle_norm)
            
        return cycles

    def _interpolate(self, array_2d):
        """将任意长度的 (6, T) 插值到 (6, target_len)"""
        original_len = array_2d.shape[1]
        x_old = np.linspace(0, 1, original_len)
        x_new = np.linspace(0, 1, self.target_len)
        
        f = interp1d(x_old, array_2d, axis=1, kind='linear')
        return f(x_new)

def convert_dataset_to_cycles(X_raw, y_raw, target_len=100):
    """
    批处理函数：将原始数据集转换为周期归一化数据集
    X_raw: (N, 12, T)
    y_raw: (N,)
    Return: X_cycles (Total_Cycles, 18, 100), y_cycles (Total_Cycles,)
    """
    processor = GaitPreprocessor(target_len=target_len, flex_ch_idx=2)
    X_list = []
    y_list = []
    
    logger.info("正在执行步态周期分割与归一化...")
    for i in range(len(X_raw)):
        cycles = processor.process_subject(X_raw[i])
        
        if cycles is not None and len(cycles) > 0:
            X_list.append(cycles)
            # 标签扩展：该样本生成的每个周期都继承原标签
            y_list.append(np.full(cycles.shape[0], y_raw[i]))
            
    if len(X_list) > 0:
        X_final = np.concatenate(X_list, axis=0)
        y_final = np.concatenate(y_list, axis=0)
        logger.info(f"周期提取完成: {X_raw.shape[0]} 样本 -> {X_final.shape[0]} 周期")
        return X_final, y_final
    else:
        logger.warning("未能提取到任何有效周期！")
        return None, None


class CycleVotingDataset(Dataset):
    """
    【投票评估专用数据集】
    仅用于测试/验证阶段。
    __getitem__ 返回的是一个样本切分出的【所有周期堆叠】，而不是单个周期。

    Return:
        cycles: Tensor [K, 12, 100] (K为该样本切分出的周期数)
        label: Tensor [1] (该样本的标签)
    """

    def __init__(self, raw_data, labels, config):
        self.raw_data = raw_data  # (N, 12, 600)
        self.labels = labels  # (N,)
        self.processor = GaitPreprocessor(target_len=100, flex_ch_idx=2)
        self.config = config

        # 预先处理好索引，因为有些样本可能提取不出周期，需要跳过
        self.valid_indices = []
        self.cached_cycles = {}  # 内存足够时缓存，加速 epoch

        logger.info("正在初始化投票数据集 (预处理周期分割)...")
        for i in range(len(raw_data)):
            # 这里我们不做左右腿配对，而是分别提取左腿和右腿的所有周期
            # 因为 Layer 1 的输入是单腿 (或者双腿拼接)
            # 假设模型输入依然是 (12, 100) -> 左+右

            # 使用 process_subject 提取成对周期 (Left+Right)
            # 输出 shape: (K, 12/18/24, 100) 取决于是否开启 Diff
            cycles = self.processor.process_subject(raw_data[i])

            if cycles is not None and len(cycles) > 0:
                self.valid_indices.append(i)

                # 同样的预处理流程 (归一化/特征工程)
                # 1. 归一化 (使用全局 scaler)
                # 注意：这里需要传入 scaler，比较麻烦。
                # 简单策略：先不做 scaler，等 getitem 时做，或者在外部做完传入
                # 为了代码简洁，假设 raw_data 已经是归一化过的 (N, 12, 600)
                # 但周期归一化会改变数值分布，最好是在周期化之后再做 scaler

                # 这里我们暂存 raw cycles，在 getitem 里做 transform
                self.cached_cycles[i] = cycles

    def __getitem__(self, index):
        real_idx = self.valid_indices[index]
        cycles = self.cached_cycles[real_idx]  # [K, 12, 100]
        label = self.labels[real_idx]

        # 应用特征工程 (Diff Channel)
        if self.config.get('use_diff_channel', True):
            # cycles: [K, 12, 100]
            # 手动调用 add_diff_channel
            # 需要适配 shape [K, 12, 100]
            cycles_diff = []
            for k in range(len(cycles)):
                # expand dim to (1, 12, 100) for function compatibility
                c = cycles[k][np.newaxis, :, :]
                c_out = add_diff_channel(c)  # -> (1, 24/36, 100)
                cycles_diff.append(c_out)
            cycles = np.concatenate(cycles_diff, axis=0)  # [K, 36, 100]

        return torch.from_numpy(cycles).float(), torch.tensor(label).long(), real_idx

    def __len__(self):
        return len(self.valid_indices)

# ==========================================
# 3. 核心处理逻辑 (Core Logic)
# ==========================================

def get_patient_label(left_label, right_label, label_type):
    """根据左右腿状态和任务类型生成标签"""
    status_map = {
        '健康': 0,
        '正常': 0,
        '韧带断裂': 1,       # ACL
        '有合并半月板损伤': 1, # ACL (包含合并)
        '无合并半月板损伤': 1, # ACL (单纯)
        '关节炎': 2,         # OA
        '无记录': 0
    }
    l_code = status_map.get(left_label, 0)
    r_code = status_map.get(right_label, 0)

    if label_type == 'ACL&OA_label':
        # 0:健康, 1:ACL, 2:OA
        if l_code == 1 or r_code == 1: return 1
        elif l_code == 2 or r_code == 2: return 2
        else: return 0
    elif label_type == 'injure_label':
        # 0:健康, 1:合并, 2:单纯, 3:OA
        if l_code == 1 or r_code == 1: return 1
        if l_code == 2 or r_code == 2: return 2
        return 0
    elif label_type == 'injure_leg_label':
        if l_code == 1: return 1 
        if r_code == 1: return 2 
        if l_code == 2: return 3 
        if r_code == 2: return 4 
        return 0
    return 0

def process_gait_dataframe(df, period_norm=False, label_type='ACL&OA_label'):
    """从DataFrame提取特征并生成 (N, 12, T) 数组"""
    samples, labels, person_ids = [], [], []
    col_name = 'norm_features' if period_norm else 'features'

    if label_type == 'injure_leg_label':
        source_col = 'injure_label'
    else:
        source_col = label_type

    if source_col not in df.columns:
        raise KeyError(f"Dataframe 中找不到列 '{source_col}'。可用的列有: {list(df.columns)}")

    groups = df.groupby('person_id')
    for person_id, group in groups:
        if len(group) != 2: continue
        group = group.sort_values('leg') 
        if group.iloc[0]['leg'] != 'left': continue 

        left_row, right_row = group.iloc[0], group.iloc[1]
        feat_l = sanitize_features_array(parse_features_json(left_row[col_name]))
        feat_r = sanitize_features_array(parse_features_json(right_row[col_name]))

        if feat_l.shape[0] > feat_l.shape[1]: 
            feat_l, feat_r = feat_l.T, feat_r.T
            
        combined = np.concatenate([feat_l, feat_r], axis=0) 
        samples.append(combined)
        lbl = get_patient_label(left_row[source_col], right_row[source_col], label_type)
        labels.append(lbl)
        person_ids.append(int(person_id))

    return np.array(samples), np.array(labels), np.array(person_ids)

# ==========================================
# 4. 特征工程与增强 (Transforms)
# ==========================================

def normalize_features_v4(config, X, is_train=True, scaler_list=None):
    num_features = 6 
    if not config.get('Norm', False):
        return X, None

    if is_train:
        scaler_list = []
        for i in range(num_features):
            left_vals = X[:, i, :].reshape(-1, 1)
            right_vals = X[:, i + num_features, :].reshape(-1, 1)
            global_vals = np.concatenate([left_vals, right_vals], axis=0)
            sc = StandardScaler()
            sc.fit(global_vals)
            scaler_list.append(sc)
            X[:, i, :] = sc.transform(left_vals).reshape(X.shape[0], -1)
            X[:, i + num_features, :] = sc.transform(right_vals).reshape(X.shape[0], -1)
    else:
        if scaler_list is None: raise ValueError("Test mode requires fitted scalers!")
        for i in range(num_features):
            sc = scaler_list[i]
            left_vals = X[:, i, :].reshape(-1, 1)
            right_vals = X[:, i + num_features, :].reshape(-1, 1)
            X[:, i, :] = sc.transform(left_vals).reshape(X.shape[0], -1)
            X[:, i + num_features, :] = sc.transform(right_vals).reshape(X.shape[0], -1)
    return X, scaler_list

def convert_to_mirror_pairs_binary(data, labels, config):
    if config.get('use_layer1_augmentation', False):
        view1_data = data.copy()
        view1_label = np.isin(labels, [1, 3]).astype(int) 
        view2_data = data.copy()
        view2_data[:, 0:6, :] = data[:, 6:12, :] 
        view2_data[:, 6:12, :] = data[:, 0:6, :]
        view2_label = np.isin(labels, [2, 4]).astype(int)
        mirror_data = np.concatenate([view1_data, view2_data], axis=0)
        mirror_labels = np.concatenate([view1_label, view2_label], axis=0)
    else:
        mirror_data = data.copy()
        mirror_labels = np.isin(labels, [1, 2, 3, 4]).astype(int)
    return mirror_data, mirror_labels

def augment_mirror_ternary(data, labels):
    mapper = np.array([0, 1, 2, 1, 2])
    mapped_labels = mapper[labels] 
    orig_data = data.copy()
    orig_labels = mapped_labels.copy()
    mirror_data = data.copy()
    mirror_data[:, 0:6, :] = data[:, 6:12, :] 
    mirror_data[:, 6:12, :] = data[:, 0:6, :]
    mirror_labels = mapped_labels.copy()
    mirror_labels[mapped_labels == 1] = 2
    mirror_labels[mapped_labels == 2] = 1
    final_data = np.concatenate([orig_data, mirror_data], axis=0)
    final_labels = np.concatenate([orig_labels, mirror_labels], axis=0)
    return final_data, final_labels

def augment_mirror_disease_type(data, labels):
    """
    【新三分类模式】生成镜像增强数据
    Task: 健康(0) / ACL(1) / OA(2)

    逻辑:
    1. 镜像翻转数据 (交换左右腿通道)
    2. 标签保持不变 (因为疾病类型不随左右腿互换而改变)
    """
    # 1. 原始数据
    orig_data = data.copy()
    orig_labels = labels.copy()

    # 2. 镜像数据 (Mirror)
    # 交换左右腿通道 [0-5] <-> [6-11]
    mirror_data = data.copy()
    mirror_data[:, 0:6, :] = data[:, 6:12, :]
    mirror_data[:, 6:12, :] = data[:, 0:6, :]

    # 标签不变! (ACL病人镜像后还是ACL病人)
    mirror_labels = labels.copy()

    # 3. 合并
    final_data = np.concatenate([orig_data, mirror_data], axis=0)
    final_labels = np.concatenate([orig_labels, mirror_labels], axis=0)

    return final_data, final_labels

def compute_derivatives(x, fs=60):
    """
    计算角加速度 (二阶导数)
    x: [C, T] 或 [B, C, T]
    fs: 采样频率 (Hz)，用于计算物理单位。默认 60Hz
    """
    # Savitzky-Golay 滤波器参数
    # window_length: 窗口长度，必须是奇数。
    # 对于 100 帧的数据，选 7 或 9 比较合适；对于 600 帧，可以选 15 或 21。
    # 这里为了自适应，根据 T 的长度动态设定
    T = x.shape[-1]
    window_length = min(11, T - 2 if T % 2 != 0 else T - 3) 
    polyorder = 2  # 多项式阶数，计算二阶导数至少需要2阶，建议2或3
    
    # deriv=2 表示计算二阶导数 (加速度)
    # delta=1.0/fs 表示时间间隔
    acc = savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=2, delta=1.0/fs, axis=-1)
    
    return acc

def add_diff_channel(X):
    """计算并拼接差值通道: L, R, Diff, AbsDiff"""
    main = X[:, 0:6, :]
    aux = X[:, 6:12, :]
    diff = main - aux
    abs_diff = np.abs(diff)
    # 返回 24 通道
    return np.concatenate([X, diff, abs_diff], axis=1)

def add_acc_channel(X):
    """计算并拼接角加速度通道"""
    # 计算角加速度 (Acceleration ~ Pseudo-Force)
    # 假设采样率 fs=60Hz (如果数据归一化到了100帧，实际上是把周期缩放了，但相对加速度波形依然有效)
    L_pos = X[:, 0:6, :]
    R_pos = X[:, 6:12, :]
    L_acc = compute_derivatives(L_pos, fs=60)
    R_acc = compute_derivatives(R_pos, fs=60)
    
    # 只加 L_acc, R_acc。Diff_acc 可以由网络层内的 subtraction 自动提取
    # 总维度 = 24 + 12 = 36
    
    combined = np.concatenate([X, L_acc, R_acc], axis=1)
    
    return combined

# ==========================================
# 5. 数据加载主控 (Main Loader)
# ==========================================

def load_gait_data(config):
    Data = {}
    
    logger.info("正在加载 CSV 数据集 ...")
    paths = {
        'train': os.path.join(config['dataset_path'], 'gait_train_A.csv'),
        'test': os.path.join(config['dataset_path'], 'gait_test_A.csv')
    }
    
    dfs = {k: pd.read_csv(v) for k, v in paths.items()}
    
    # 1. 读取原始长序列 (N, 12, 600)
    x_train, y_train, _ = process_gait_dataframe(dfs['train'], False, config['label_type'])
    x_test, y_test, _ = process_gait_dataframe(dfs['test'], False, config['label_type'])

    if config.get('use_cycle_segmentation', False):
        logger.info("启用周期分割 (Cycle Segmentation): 正在从原始数据提取周期...")
        x_train_cycles, y_train_cycles = convert_dataset_to_cycles(x_train, y_train, target_len=100)
        x_test_cycles, y_test_cycles = convert_dataset_to_cycles(x_test, y_test, target_len=100)
        
        # 将生成的周期数据存入 Data 字典
        Data['train_data_b'], Data['train_label_b'] = x_train_cycles, y_train_cycles
        Data['test_data_b'], Data['test_label_b'] = x_test_cycles, y_test_cycles
        x_train, y_train = x_train_cycles, y_train_cycles
        if not config.get('cycle_voting', True):
            x_test, y_test = x_test_cycles, y_test_cycles

        logger.info(f"周期数据生成完毕")
    
    Data['train_data_a'] = x_train
    Data['train_label_a'] = y_train
    Data['test_data_a'] = x_test
    Data['test_label_a'] = y_test

    # 3. Layer 2 数据准备 (仅患病样本)
    train_mask_sick = (y_train != 0)
    test_mask_sick = (y_test != 0)
    Data['train_data_layer2'] = x_train[train_mask_sick]
    Data['test_data_layer2'] = x_test[test_mask_sick]
    
    def map_layer2_label(y):
        new_y = np.zeros_like(y)
        new_y[np.isin(y, [1, 2])] = 0 
        new_y[np.isin(y, [3, 4])] = 1 
        return new_y

    Data['train_label_layer2'] = map_layer2_label(y_train[train_mask_sick])
    Data['test_label_layer2'] = map_layer2_label(y_test[test_mask_sick])

    # 4. Layer 1 数据准备
    Data['train_data_layer1'] = x_train
    Data['train_label_layer1'] = y_train
    Data['test_data_layer1'] = x_test
    Data['test_label_layer1'] = y_test

    # 5. 归一化 (Normalization)
    Data['train_data_layer1'], scaler = normalize_features_v4(config, Data['train_data_layer1'], is_train=True)
    Data['test_data_layer1'], _ = normalize_features_v4(config, Data['test_data_layer1'], is_train=False, scaler_list=scaler)
    
    Data['train_data_layer2'], scaler_l2 = normalize_features_v4(config, Data['train_data_layer2'], is_train=True)
    Data['test_data_layer2'], _ = normalize_features_v4(config, Data['test_data_layer2'], is_train=False, scaler_list=scaler_l2)

    # 6. 镜像与增强 (Mirror Pairing)
    task_type = config.get('layer1_task', 'binary')
    logger.info(f"Layer 1 Task: {task_type}")
    
    if task_type == 'binary':
        Data['train_data_layer1'], Data['train_label_layer1'] = convert_to_mirror_pairs_binary(
            Data['train_data_layer1'], Data['train_label_layer1'], config
        )
        Data['test_data_layer1'], Data['test_label_layer1'] = convert_to_mirror_pairs_binary(
            Data['test_data_layer1'], Data['test_label_layer1'], config
        )
    else:
        # Ternary 模式强制使用镜像增强
        Data['train_data_layer1'], Data['train_label_layer1'] = augment_mirror_disease_type(
            Data['train_data_layer1'], Data['train_label_layer1']
        )
        Data['test_data_layer1'], Data['test_label_layer1'] = augment_mirror_disease_type(
            Data['test_data_layer1'], Data['test_label_layer1']
        )

    # 7. 特征工程 (Diff + AbsDiff)
    if config.get('use_diff_channel', True):
        logger.info("启用 Diff 通道增强: 输入维度 -> 24")
        Data['train_data_layer1'] = add_diff_channel(Data['train_data_layer1'])
        Data['test_data_layer1'] = add_diff_channel(Data['test_data_layer1'])

        Data['train_data_layer2'] = add_diff_channel(Data['train_data_layer2'])
        Data['test_data_layer2'] = add_diff_channel(Data['test_data_layer2'])
    else:
        logger.info("禁用 Diff 通道增强")
    
    # 8. Layer 2 Augmentation
    if config.get('use_layer2_augmentation', False):
        augment_config = {
            'jitter_std': 0.02,
            'scale_range': [0.90, 1.10],
            'rotation_range': [-5, 5],
            'time_warp_knots': 4,
            'magnitude_warp_std': 0.15
        }
        Data['train_data_layer2'], Data['train_label_layer2'] = augment_layer2_data(
            Data['train_data_layer2'], 
            Data['train_label_layer2'],
            n_augments=config.get('n_augments', 10),
            config=augment_config
        )

    # 9. Joint 数据集
    Data['train_dataset_joint'] = single_branch_dataset(Data['train_data_a'], Data['train_label_a']) 
    Data['test_dataset_joint'] = single_branch_dataset(Data['test_data_a'], Data['test_label_a'])

    logger.info(f"layer1训练集: {Data['train_data_layer1'].shape}, layer1测试集: {Data['test_data_layer1'].shape}")
    logger.info(f"layer2训练集: {Data['train_data_layer2'].shape}, layer2测试集: {Data['test_data_layer2'].shape}")
    return Data

# 简单的 Dataset 类定义
from torch.utils.data import Dataset
import torch

class single_branch_dataset(Dataset):
    def __init__(self, data, label):
        self.feature = data.astype(np.float32)
        self.labels = label.astype(np.int64)
    def __getitem__(self, index):
        return torch.tensor(self.feature[index]), torch.tensor(self.labels[index]), index
    def __len__(self):
        return len(self.labels)