# modules/processor.py
import numpy as np
import pandas as pd
from scipy.signal import kaiserord, firwin, filtfilt, butter
from utils import parse_features_to_array

class SignalProcessor:
    def __init__(self, fs=60, params=None):
        self.fs = fs
        self.params = params or {}

    def _notch_filter(self, data):
        nyq = self.fs / 2
        freq = self.params.get('notch_freq', 10)
        width = self.params.get('width', 2)
        ripple = self.params.get('ripple_db', 60)
        
        low = max(0.01, freq - width/2)
        high = min(nyq - 0.01, freq + width/2)
        
        N, beta = kaiserord(ripple, width / nyq)
        if N % 2 == 0: N += 1
        taps = firwin(N, [low/nyq, high/nyq], window=('kaiser', beta), pass_zero='bandstop')
        return filtfilt(taps, 1.0, data, axis=0)

    def _lowpass_filter(self, data):
        nyq = self.fs / 2
        cutoff = self.params.get('cutoff', 6)
        order = self.params.get('order', 6)
        b, a = butter(order, cutoff / nyq, btype='low')
        return filtfilt(b, a, data, axis=0)

    def process_features(self, features_array):
        """应用滤波"""
        filtered = self._notch_filter(features_array)
        filtered = self._lowpass_filter(filtered)
        return filtered

class DataCleaner:
    @staticmethod
    def remove_no_record(df):
        """删除无记录的样本并重置 person_id"""
        print(f"清洗前样本数: {len(df)}")
        df_clean = df[df['injure_label'] != '无记录'].copy()
        # 重置 person_id: 每两个人(左+右)为一个ID
        # 注意：必须先reset_index确保顺序
        df_clean.reset_index(drop=True, inplace=True)
        new_ids = (np.arange(len(df_clean)) // 2) + 1
        df_clean['person_id'] = new_ids.astype(int)
        print(f"清洗后样本数: {len(df_clean)}")
        return df_clean

    @staticmethod
    def fix_shape_and_filter(df, processor):
        """修正形状 (padding) 并应用滤波"""
        fixed_count = 0
        processed_features = []
        
        for idx, row in df.iterrows():
            features = parse_features_to_array(row['features'])
            
            # 1. 形状修正 (Padding)
            target_len = 600
            if features.shape[0] < target_len:
                missing = target_len - features.shape[0]
                padding = np.tile(features[-1, :], (missing, 1))
                features = np.vstack([features, padding])
                fixed_count += 1
            
            # 2. 信号处理 (Filtering)
            features = processor.process_features(features)
            processed_features.append(features.tolist())
            
        df['features'] = processed_features
        print(f"已修正(Padding) {fixed_count} 条数据形状。")
        return df

    @staticmethod
    def drop_invalid_samples(df):
        """
        检查并删除包含 NaN 或 Inf 的样本。
        【安全策略】如果某人的其中一条腿数据无效，则删除该人的所有数据（确保左右腿成对）。
        """
        initial_count = len(df)
        
        def is_valid(features_list):
            # 转为 numpy 数组检查
            arr = np.array(features_list)
            return not (np.isnan(arr).any() or np.isinf(arr).any())

        # 1. 初步筛选：找出哪些行是无效的
        # is_valid_row 是一个 boolean Series (True=有效, False=无效)
        is_valid_row = df['features'].apply(is_valid)
        
        # 2. 找出"坏人"：哪些 person_id 拥有至少一条无效数据
        # 取反 is_valid_row 得到无效行，然后提取这些行的 person_id
        invalid_person_ids = df[~is_valid_row]['person_id'].unique()
        
        if len(invalid_person_ids) > 0:
            print(f"⚠️ 发现 {len(invalid_person_ids)} 个受试者存在无效数据(NaN/Inf)。")
            print(f"   受影响 ID: {list(invalid_person_ids)}")
            
            # 3. 连坐策略：剔除这些人的所有数据（包括他们的健康腿）
            # 筛选出 person_id 不在 invalid_person_ids 中的行
            df_clean = df[~df['person_id'].isin(invalid_person_ids)].copy()
            
            dropped_count = initial_count - len(df_clean)
            print(f"🚨以此触发成对删除策略：共剔除 {dropped_count} 条样本（确保受影响ID的左右腿完全移除）。")
            
            # 重新 reset_index 以防止索引断层
            df_clean.reset_index(drop=True, inplace=True)
            return df_clean
            
        else:
            print("✅ 未发现无效样本，数据集完整。")
            return df