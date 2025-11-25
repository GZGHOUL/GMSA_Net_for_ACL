import os
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import json, re
from imblearn.over_sampling import SMOTE
from Dataset.gait_augmentation import augment_layer2_data, augment_layer1_data


logger = logging.getLogger(__name__)

def parse_features_json(s: str) -> np.ndarray:
    # 将各种大小写的 nan/NaN/NAN 替换成 JSON 的 null
    s = re.sub(r'\bnan\b', 'null', str(s), flags=re.IGNORECASE)
    data = json.loads(s)                  # -> list/list of lists
    arr = np.array(data, dtype=float)
    # json 的 null 会变成 np.nan 自动兼容
    return arr

def sanitize_features_array(arr: np.ndarray, clip_value: float = None) -> np.ndarray:
    """Replace NaN/Inf with finite numbers and optionally clip extreme values.
    - NaN -> 0.0
    - +Inf -> large finite value
    - -Inf -> large negative finite value
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=float)
    # Use a conservative large value based on dtype
    pos_limit = 1e6 if clip_value is None else float(clip_value)
    neg_limit = -pos_limit
    arr = np.nan_to_num(arr, nan=0.0, posinf=pos_limit, neginf=neg_limit)
    if clip_value is not None:
        np.clip(arr, neg_limit, pos_limit, out=arr)
    return arr

def process_gait_data(df, period_norm=False, label_type='ACL_label'):
    """处理步态数据：将每个人的两条腿数据合并为一个多元时间序列样本"""
    samples = []
    labels = []
    person_ids = []

    for person_id, group in df.groupby('person_id'):
        if len(group) != 2:
            continue

        # 按腿排序确保顺序一致
        group = group.sort_values('leg')
        left_leg = group[group['leg'] == 'left'].iloc[0]
        right_leg = group[group['leg'] == 'right'].iloc[0]

         # 提取特征数据
        if period_norm:
            left_features = parse_features_json(left_leg['norm_features'])  # [时间步长, 6]
            right_features = parse_features_json(right_leg['norm_features'])  # [时间步长, 6]
        else:
            left_features = parse_features_json(left_leg['features'])  # [时间步长, 6]
            right_features = parse_features_json(right_leg['features'])  # [时间步长, 6]

        # 数值清洗，移除 NaN/Inf，避免后续网络/损失出现 NaN
        left_features = sanitize_features_array(left_features)
        right_features = sanitize_features_array(right_features)
            
        # 合并为多元时间序列: [12个变量 (左腿6维+右腿6维), 时间步长]
        # ShapeFormer期望格式: [n_variables, sequence_length]
        combined_features = np.concatenate([left_features, right_features], axis=1)  # [时间步长, 12]
        combined_features = combined_features.T  # [12, 时间步长]
        # combined_features = sanitize_features_array(combined_features)
            
        samples.append(combined_features)
            
        # 获取患者标签
        if label_type == 'ACL_label':
            left_label = left_leg['ACL_label']
            right_label = right_leg['ACL_label']
             # 三分类标签：0=健康, 1=左腿断裂, 2=右腿断裂
            if left_label == '健康' and right_label == '健康':
                patient_label = 0
            elif left_label == '韧带断裂' and right_label == '健康':
                patient_label = 1
            elif left_label == '健康' and right_label == '韧带断裂':
                patient_label = 2
            else:
                patient_label = 0  # 默认为健康

        elif label_type == 'injure_label':
            left_label = left_leg['injure_label']
            right_label = right_leg['injure_label']
            # 三分类标签：0=健康, 1=有合并半月板损伤, 2=无合并半月板损伤
            if left_label == '健康' and right_label == '健康':
                patient_label = 0
            elif left_label == '有合并半月板损伤' and right_label == '健康':
                patient_label = 1
            elif left_label == '健康' and right_label == '有合并半月板损伤':
                patient_label = 1
            elif left_label == '无合并半月板损伤' and right_label == '健康':
                patient_label = 2
            elif left_label == '健康' and right_label == '无合并半月板损伤':
                patient_label = 2
            else:
                patient_label = 0  # 默认为健康

        elif label_type == 'injure_leg_label':
            left_label = left_leg['injure_label']
            right_label = right_leg['injure_label']
            # 标签：0=健康， 1=左腿有合并半月板损伤，2=右腿有合并半月板损伤，3=左腿无合并半月板损伤，4=右腿无合并半月板损伤
            if left_label == '健康' and right_label == '健康':
                patient_label = 0
            elif left_label == '有合并半月板损伤' and right_label == '健康':
                patient_label = 1
            elif left_label == '健康' and right_label == '有合并半月板损伤':
                patient_label = 2
            elif left_label == '无合并半月板损伤' and right_label == '健康':
                patient_label = 3
            elif left_label == '健康' and right_label == '无合并半月板损伤':
                patient_label = 4
            else:
                patient_label = 0  # 默认为健康
        
        labels.append(patient_label)
        person_ids.append(int(person_id))

    return np.array(samples), np.array(labels), np.array(person_ids)

def normalize_features(config, X, is_train=True, scaler=None):
    if config.get('Norm', False):
        if is_train:
            scaler = []
            for var_idx in range(X.shape[1]):
                sc = StandardScaler()
                train_var_data = X[:, var_idx, :].reshape(-1, 1)
                sc.fit(train_var_data)
                scaler.append(sc)

                X[:, var_idx, :] = sc.transform(X[:, var_idx, :].reshape(-1, 1)).reshape(X.shape[0], -1)
        else:
            for var_idx, sc in enumerate(scaler):
                X[:, var_idx, :] = sc.transform(X[:, var_idx, :].reshape(-1, 1)).reshape(X.shape[0], -1)

        # 归一化后再次保障数值稳定
        X = sanitize_features_array(X)
    return X, scaler

def normalize_features_v2(config, X, is_train=True, scaler=None):
    if config.get('Norm', False):
        if is_train:
            scaler = []
            sc = StandardScaler()
            for i in range(3):
                train_var_data = X[:, i*3:(i+1)*3, :].reshape(-1, 1)
                sc.fit(train_var_data)
                X[:, i*3:(i+1)*3, :] = sc.transform(X[:, i*3:(i+1)*3, :].reshape(-1, 1)).reshape(X.shape[0], 3, -1)
                scaler.append(sc)
        else:
            for i in range(3):
                X[:, i*3:(i+1)*3, :] = scaler[i].transform(X[:, i*3:(i+1)*3, :].reshape(-1, 1)).reshape(X.shape[0], 3, -1)
    return X, scaler

def normalize_features_v3(config, X, is_train=True, scaler=None):
    if config.get('Norm', False):
        if is_train:
            scaler = []
            sc = StandardScaler()
            for i in range(1):
                train_var_data_left = X[:, i*3:(i+1)*3, :].reshape(-1, 1)
                train_var_data_right = X[:, (i+2)*3:(i+3)*3, :].reshape(-1, 1)
                train_var_data = np.append(train_var_data_left, train_var_data_right, axis=0)
                sc.fit(train_var_data)
                X[:, i*3:(i+1)*3, :] = sc.transform(X[:, i*3:(i+1)*3, :].reshape(-1, 1)).reshape(X.shape[0], 3, -1)
                X[:, (i+2)*3:(i+3)*3, :] = sc.transform(X[:, (i+2)*3:(i+3)*3, :].reshape(-1, 1)).reshape(X.shape[0], 3, -1)
                scaler.append(sc)
        else:
            for i in range(1):
                X[:, i*3:(i+1)*3, :] = scaler[i].transform(X[:, i*3:(i+1)*3, :].reshape(-1, 1)).reshape(X.shape[0], 3, -1)
                X[:, (i+2)*3:(i+3)*3, :] = scaler[i].transform(X[:, (i+2)*3:(i+3)*3, :].reshape(-1, 1)).reshape(X.shape[0], 3, -1)
    return X, scaler

def normalize_features_v4(config, X, is_train=True, scaler_list=None):
    num_features = 6
    if config.get('Norm', False):
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

            X[:, 12:, :] = X[:, 0:6, :] - X[:, 6:12, :]

        else:
            if scaler_list is None:
                raise ValueError("测试阶段必须提供训练好的 scaler_list！")
                
            for i in range(num_features):
                sc = scaler_list[i]
                
                # 提取
                left_vals = X[:, i, :].reshape(-1, 1)
                right_vals = X[:, i + num_features, :].reshape(-1, 1)
                
                # 变换 (只 Transform, 不 Fit)
                X[:, i, :] = sc.transform(left_vals).reshape(X.shape[0], -1)
                X[:, i + num_features, :] = sc.transform(right_vals).reshape(X.shape[0], -1)
            
            # 重算 Diff
            X[:, 12:, :] = X[:, 0:6, :] - X[:, 6:12, :]

    return X, scaler_list

def balance_layer2_data(train_data, train_label):
    """使用 SMOTE 生成合成样本"""
    # 将 [N, 12, T] 展平为 [N, 12*T]
    n_samples, n_dims, n_timesteps = train_data.shape
    X_flat = train_data.reshape(n_samples, -1)
    
    # SMOTE 过采样
    smote = SMOTE(
        sampling_strategy='minority',  # 只过采样少数类
        k_neighbors=3,  # 少数类样本少，减少邻居数
        random_state=42
    )
    X_resampled, y_resampled = smote.fit_resample(X_flat, train_label)
    
    # 恢复形状
    X_resampled = X_resampled.reshape(-1, n_dims, n_timesteps)
    
    return X_resampled, y_resampled

def split_layer_data(config, Data):
    Data['train_data_layer1'] = Data['train_data_a']
    Data['train_label_layer1'] = []  

    Data['train_data_layer2'] = []   
    Data['train_label_layer2'] = []

    Data['test_data_layer1'] = Data['test_data_a']
    Data['test_label_layer1'] = []

    Data['test_data_layer2'] = []
    Data['test_label_layer2'] = []

    if config['label_type'] == 'injure_label':
        for label in Data['train_label_a']:
            if label == 0:
                Data['train_label_layer1'].append(0)
            else:
                Data['train_label_layer1'].append(1)

        for label in Data['test_label_a']:
            if label == 0:
                Data['test_label_layer1'].append(0)
            else:
                Data['test_label_layer1'].append(1)

        for idx, label in enumerate(Data['train_label_a']):
            if label == 1:
                Data['train_data_layer2'].append(Data['train_label_a'][idx])
                Data['train_label_layer2'].append(0)
            elif label == 2:
                Data['train_data_layer2'].append(Data['train_label_a'][idx])
                Data['train_label_layer2'].append(1)

        for idx, label in enumerate(Data['test_label_a']):
            if label == 1:
                Data['test_data_layer2'].append(Data['test_data_a'][idx])
                Data['test_label_layer2'].append(0)
            elif label == 2:
                Data['test_data_layer2'].append(Data['test_data_a'][idx])
                Data['test_label_layer2'].append(1)
    
    if config['label_type'] == 'injure_leg_label':
        # 第一层标签：0=健康，1=左腿患病，2=右腿患病
        for label in Data['train_label_a']:
            if label == 0:
                Data['train_label_layer1'].append(0)
            elif label == 1 or label == 3:
                Data['train_label_layer1'].append(1)
            elif label == 2 or label == 4:
                Data['train_label_layer1'].append(2)

        for label in Data['test_label_a']:
            if label == 0:
                Data['test_label_layer1'].append(0)
            elif label == 1 or label == 3:
                Data['test_label_layer1'].append(1)
            elif label == 2 or label == 4:
                Data['test_label_layer1'].append(2)

        for idx, label in enumerate(Data['train_label_a']):
            if label == 1 or label == 2:
                Data['train_data_layer2'].append(Data['train_data_a'][idx])
                Data['train_label_layer2'].append(0)
            elif label == 3 or label == 4:
                Data['train_data_layer2'].append(Data['train_data_a'][idx])
                Data['train_label_layer2'].append(1)

        for idx, label in enumerate(Data['test_label_a']):
            if label == 1 or label == 2:
                Data['test_data_layer2'].append(Data['test_data_a'][idx])
                Data['test_label_layer2'].append(0)
            elif label == 3 or label == 4:
                Data['test_data_layer2'].append(Data['test_data_a'][idx])
                Data['test_label_layer2'].append(1)
    return Data

def load_gait_data(config):
    Data = {}
    # 读取训练集和测试集
    train_path = os.path.join(config['dataset_path'], 'gait_train_A.csv')
    test_path = os.path.join(config['dataset_path'], 'gait_test_A.csv')
    period_norm_train_path = os.path.join(config['dataset_path'], 'gait_train_B.csv')
    period_norm_test_path = os.path.join(config['dataset_path'], 'gait_test_B.csv')

    logger.info("正在加载步态数据集 ...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    period_norm_train_df = pd.read_csv(period_norm_train_path)
    period_norm_test_df = pd.read_csv(period_norm_test_path)

    # 处理训练集和测试集
    x_train, y_train, _ = process_gait_data(train_df, period_norm=False, label_type=config['label_type'])
    x_test, y_test, _ = process_gait_data(test_df, period_norm=False, label_type=config['label_type'])
    x_period_norm_train, y_period_norm_train, _ = process_gait_data(period_norm_train_df, period_norm=True, label_type=config['label_type'])
    x_period_norm_test, y_period_norm_test, _ = process_gait_data(period_norm_test_df, period_norm=True, label_type=config['label_type'])

    x_train_augment = np.load(os.path.join(config['dataset_path'], 'x_train_augment.npy'))
    y_train_augment = np.load(os.path.join(config['dataset_path'], 'y_train_augment.npy'))
    x_test_augment = np.load(os.path.join(config['dataset_path'], 'x_test_augment.npy'))
    y_test_augment = np.load(os.path.join(config['dataset_path'], 'y_test_augment.npy'))

    logger.info(f"训练集数据形状: {x_train_augment.shape}, 测试集数据形状: {x_test_augment.shape}, 类别数量：{len(np.unique(y_train_augment))}")

    Data['train_data_a'] = x_train_augment
    Data['train_label_a'] = y_train_augment

    Data['test_data_a'] = x_test_augment
    Data['test_label_a'] = y_test_augment

    Data = split_layer_data(config, Data)

    Data['train_data_layer1'] = np.array(Data['train_data_layer1'])
    Data['train_label_layer1'] = np.array(Data['train_label_layer1'])

    Data['train_data_layer2'] = np.array(Data['train_data_layer2'])
    Data['train_label_layer2'] = np.array(Data['train_label_layer2'])

    Data['test_data_layer1'] = np.array(Data['test_data_layer1'])
    Data['test_label_layer1'] = np.array(Data['test_label_layer1'])

    Data['test_data_layer2'] = np.array(Data['test_data_layer2'])
    Data['test_label_layer2'] = np.array(Data['test_label_layer2'])
    
    Data['train_data_layer1'], scaler_layer2 = normalize_features_v4(config, Data['train_data_layer1'], is_train=True)
    Data['test_data_layer1'], _ = normalize_features_v4(config, Data['test_data_layer1'], is_train=False, scaler_list=scaler_layer2)

    Data['train_data_layer2'], scaler_layer2 = normalize_features_v2(config, Data['train_data_layer2'], is_train=True)
    Data['test_data_layer2'], _ = normalize_features_v2(config, Data['test_data_layer2'], is_train=False, scaler=scaler_layer2)

    if config.get('use_augmentation', False):
        new_data, new_labels = augment_layer1_data(Data['train_data_layer1'], Data['train_label_layer1'])
        Data['train_data_layer1'] = np.concatenate((Data['train_data_layer1'], new_data), axis=0)
        Data['train_label_layer1'] = np.concatenate((Data['train_label_layer1'], new_labels), axis=0)

    if config.get('use_augmentation', False):
        logger.info("=" * 50)
        logger.info("开始数据增强...")
        logger.info(f"增强前 Layer2: {np.bincount(Data['train_label_layer2'])}")
        
        augment_config = {
            'jitter_std': 0.02,           # 较小的噪声
            'scale_range': [0.95, 1.05],  # 较小的缩放
            'rotation_range': [-3, 3],    # 较小的旋转
            'time_warp_knots': 4,
            'magnitude_warp_std': 0.15,
            'smooth_sigma': 1.0
        }
        
        Data['train_data_layer2'], Data['train_label_layer2'] = augment_layer2_data(
            Data['train_data_layer2'], 
            Data['train_label_layer2'],
            n_augments=config.get('n_augments', 10),  # 每个样本生成5个增强样本
            config=augment_config
        )
        
        logger.info(f"增强后 Layer2: {np.bincount(Data['train_label_layer2'])}")
        logger.info("数据增强完成！")
        logger.info("=" * 50)

    Data['train_data_b'] = x_period_norm_train
    Data['train_label_b'] = y_period_norm_train
    Data['test_data_b'] = x_period_norm_test
    Data['test_label_b'] = y_period_norm_test


    return Data