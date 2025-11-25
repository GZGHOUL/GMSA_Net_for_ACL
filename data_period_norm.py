import numpy as np
import os
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import tqdm
from Dataset.data_loader import process_gait_data
from hierarchical_main import get_arg
from utils import Setup
import pandas as pd

class GaitPreprocessor:
    def __init__(self, target_len=100, flex_ch_idx=2):
        """
        target_len: 归一化后的步态周期长度 (推荐 100)
        flex_ch_idx: 屈伸角所在的通道索引 (根据你的描述，假设是第3个通道，即索引2)
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
            
            # 计算差值特征 (关键步骤：捕捉不对称性)
            Diff = L - R        # (6, 100)
            
            # 堆叠: 通道维变为 6+6+6=18
            combined = np.concatenate([L, R, Diff], axis=0) # (18, 100)
            processed_samples.append(combined)

        return np.array(processed_samples)

    def _extract_and_normalize(self, leg_data):
        """内部函数：单腿周期提取"""
        flexion = leg_data[self.flex_idx, :]
        
        # 寻找波峰 (Peak Detection)
        # height: 设定最小高度防止噪声
        # distance: 设定波峰最小间距 (600帧里大概4-6个周期，间距设为60-80比较安全)
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

# 定义处理并收集数据的函数
def process_and_collect(subject_list, subset_name):
    X_list = []
    y_list = []
    processor = GaitPreprocessor(target_len=100, flex_ch_idx=2)
    print(f"正在处理 {subset_name} ...")
    for subj in tqdm.tqdm(subject_list):
        # 1. 提取该病人的所有周期
        cycles = processor.process_subject(subj['data'])
            
        if cycles is not None and len(cycles) > 0:
            # 2. 收集数据 (N_cycles, 18, 100)
            X_list.append(cycles)
                
            # 3. 标签对齐 (将同一个Label赋给该病人的所有周期)
            # 创建一个形状为 (N_cycles,) 的标签数组
            labels = np.full(cycles.shape[0], subj['label'])
            y_list.append(labels)
        
    if len(X_list) > 0:
        X_final = np.concatenate(X_list, axis=0)
        y_final = np.concatenate(y_list, axis=0)
        return X_final, y_final
    else:
        return None, None


def main():
    config = Setup(get_arg())
    # 读取训练集和测试集
    train_path = os.path.join(config['dataset_path'], 'gait_train_A.csv')
    test_path = os.path.join(config['dataset_path'], 'gait_test_A.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    x_train, y_train, person_ids_train = process_gait_data(train_df, period_norm=False, label_type=config['label_type'])
    x_test, y_test, person_ids_test = process_gait_data(test_df, period_norm=False, label_type=config['label_type'])

    subjects_train = []
    for i in range(len(x_train)):
        subjects_train.append({
            'data': x_train[i],  # 单个病人的 (12, 600) 数据
            'label': y_train[i]  # 单个病人的标签
        })

    subjects_test = []
    for i in range(len(x_test)):
        subjects_test.append({
            'data': x_test[i],
            'label': y_test[i]
        })

    # --- 执行处理 ---
    x_train_augment, y_train_augment = process_and_collect(subjects_train, "Train Set")
    x_test_augment, y_test_augment = process_and_collect(subjects_test, "Test Set")

    save_dir = 'Dataset/Gait2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if x_train_augment is not None:
        print(f"\n处理完成!")
        print(f"训练集样本数 (Cycles): {x_train_augment.shape} | 标签数: {y_train_augment.shape}")
        print(f"测试集样本数 (Cycles): {x_test_augment.shape}  | 标签数: {y_test_augment.shape}")
        
        # 保存为 .npy (推荐)
        np.save(os.path.join(save_dir, 'x_train_augment.npy'), x_train_augment)
        np.save(os.path.join(save_dir, 'y_train_augment.npy'), y_train_augment)
        np.save(os.path.join(save_dir, 'x_test_augment.npy'), x_test_augment)
        np.save(os.path.join(save_dir, 'y_test_augment.npy'), y_test_augment)
        
        print(f"文件已保存至 {save_dir} (.npy格式)")
    else:
        print("错误：未能提取到有效数据，请检查原始数据格式或波峰检测参数。")

if __name__ == '__main__':
    main()

