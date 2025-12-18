# modules/splitter.py
import pandas as pd
import numpy as np

class DatasetSplitter:
    def __init__(self, seed=42, train_ratio=0.7):
        self.seed = seed
        self.train_ratio = train_ratio

    def split_and_save(self, df_health, df_patient, output_paths):
        # 获取所有 person_id
        health_ids = df_health['person_id'].unique()
        patient_ids = df_patient['person_id'].unique()

        # 随机打乱
        np.random.seed(self.seed)
        health_ids = np.random.permutation(health_ids)
        patient_ids = np.random.permutation(patient_ids)

        # 划分 ID
        n_h_train = int(len(health_ids) * self.train_ratio)
        n_p_train = int(len(patient_ids) * self.train_ratio)

        h_train_ids = health_ids[:n_h_train]
        h_test_ids = health_ids[n_h_train:]
        p_train_ids = patient_ids[:n_p_train]
        p_test_ids = patient_ids[n_p_train:]

        # 筛选数据
        df_train = pd.concat([
            df_health[df_health['person_id'].isin(h_train_ids)],
            df_patient[df_patient['person_id'].isin(p_train_ids)]
        ], ignore_index=True)

        df_test = pd.concat([
            df_health[df_health['person_id'].isin(h_test_ids)],
            df_patient[df_patient['person_id'].isin(p_test_ids)]
        ], ignore_index=True)

        # 重新编号 ID
        df_train = self._renum_id(df_train)
        df_test = self._renum_id(df_test)

        # 拆分 A/B 集并保存
        self._save_split(df_train, output_paths['train_A'], output_paths['train_B'])
        self._save_split(df_test, output_paths['test_A'], output_paths['test_B'])
        
        print("数据集划分完成并保存。")

    def _renum_id(self, df):
        new_ids = (np.arange(len(df)) // 2) + 1
        df['person_id'] = new_ids.astype(int)
        return df

    def _save_split(self, df, path_a, path_b):
        cols_common = ['person_id', 'leg', 'ACL_label', 'injure_label']
        
        df_a = df[cols_common + ['features']]
        df_b = df[cols_common + ['norm_features']]
        
        df_a.to_csv(path_a, index=False)
        df_b.to_csv(path_b, index=False)