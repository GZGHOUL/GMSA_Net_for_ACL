# modules/builder.py
import pandas as pd
import numpy as np
import os
from utils import read_csv_with_fallback

class DatasetBuilder:
    def __init__(self, xlsx_paths, label_path):
        self.xlsx_paths = xlsx_paths
        self.label_path = label_path
        self.label_df = None

    def load_labels(self):
        self.label_df = read_csv_with_fallback(self.label_path)

    def _extract_leg_data(self, df, start_row, end_row, col_start, col_end):
        """提取并插值单腿数据"""
        features = df.iloc[start_row:end_row, col_start:col_end].astype(float)
        features = features.interpolate(method='linear', axis=0, limit_direction='both')
        features = features.fillna(method='bfill').fillna(method='ffill')
        return features.values.tolist()

    def build_health_dataset(self):
        print("正在构建健康人数据集...")
        excel = pd.ExcelFile(self.xlsx_paths['Health333'], engine='openpyxl')
        data = []
        
        for idx in range(0, len(excel.sheet_names), 2):
            person_id = idx // 2 + 1
            left_sheet = excel.sheet_names[idx]
            right_sheet = excel.sheet_names[idx + 1]
            
            df_left = pd.read_excel(excel, sheet_name=left_sheet, header=None)
            df_right = pd.read_excel(excel, sheet_name=right_sheet, header=None)

            # 提取特征
            feat_l = self._extract_leg_data(df_left, 9, 609, 2, 8)
            norm_l = self._extract_leg_data(df_left, 9, 109, 44, 50)
            feat_r = self._extract_leg_data(df_right, 9, 609, 23, 29)
            norm_r = self._extract_leg_data(df_right, 9, 109, 65, 71)

            # 添加记录 (双腿均为健康)
            data.append({'person_id': person_id, 'leg': 'left', 'features': feat_l, 'norm_features': norm_l, 
                         'ACL_label': '健康', 'injure_label': '健康'})
            data.append({'person_id': person_id, 'leg': 'right', 'features': feat_r, 'norm_features': norm_r, 
                         'ACL_label': '健康', 'injure_label': '健康'})
            
        return pd.DataFrame(data)

    def build_patient_dataset(self):
        print("正在构建患者数据集...")
        if self.label_df is None:
            self.load_labels()
            
        data = []
        
        # 处理函数：复用逻辑
        def process_file(excel_path, start_pid_offset, sheet_start_idx=0):
            excel = pd.ExcelFile(excel_path, engine='openpyxl')
            # ACL214 从 index 4 开始 (dataset_build.py 原逻辑)
            # ACL109 从 index 0 开始
            
            for idx in range(sheet_start_idx, len(excel.sheet_names), 2):
                # 计算 person_id (这是原代码中的硬编码逻辑，保留以确保一致性)
                if 'ACL214' in excel_path:
                    pid = (idx - 4) // 2 + 1
                else: # ACL109
                    pid = idx // 2 + 1 + 212
                
                left_sheet = excel.sheet_names[idx]
                right_sheet = excel.sheet_names[idx + 1]
                
                df_left = pd.read_excel(excel, sheet_name=left_sheet, header=None)
                df_right = pd.read_excel(excel, sheet_name=right_sheet, header=None)

                # 获取标签
                acl_status = 'RB' if df_left.iloc[1, 7] == '正常' else 'LB'
                injure_status_code = self.label_df.iloc[pid - 1, 17]
                
                injure_map = {0: 'no_injure', 1: 'injure'}
                injure_status = injure_map.get(injure_status_code, 'no_record')

                # 确定每条腿的标签
                labels = self._determine_labels(acl_status, injure_status)

                # 提取数据
                feat_l = self._extract_leg_data(df_left, 9, 609, 2, 8)
                norm_l = self._extract_leg_data(df_left, 9, 109, 44, 50)
                feat_r = self._extract_leg_data(df_right, 9, 609, 23, 29)
                norm_r = self._extract_leg_data(df_right, 9, 109, 65, 71)

                data.append({'person_id': pid, 'leg': 'left', 'features': feat_l, 'norm_features': norm_l, 
                             'ACL_label': labels['L_ACL'], 'injure_label': labels['L_Injure']})
                data.append({'person_id': pid, 'leg': 'right', 'features': feat_r, 'norm_features': norm_r, 
                             'ACL_label': labels['R_ACL'], 'injure_label': labels['R_Injure']})

        # 处理两个文件
        process_file(self.xlsx_paths['ACL214'], 0, sheet_start_idx=4)
        process_file(self.xlsx_paths['ACL109'], 212, sheet_start_idx=0)
        
        return pd.DataFrame(data)

    def build_oa_dataset(self):
        print("正在构建关节炎(OA)患者数据集...")
        data = []

        # 定义要处理的文件列表
        oa_files = ['OA72', 'OA60']

        # 为了避免 ID 冲突，OA 患者 ID 从 1000 开始编号
        current_pid = 1000

        for key in oa_files:
            file_path = self.xlsx_paths[key]
            if not os.path.exists(file_path):
                print(f"⚠️警告: 未找到文件 {file_path}，跳过。")
                continue

            print(f"处理文件: {key} ...")
            excel = pd.ExcelFile(file_path, engine='openpyxl')

            # 遍历所有 Sheet (每两个 Sheet 为一个人)
            for idx in range(1, len(excel.sheet_names), 2):
                if idx + 1 >= len(excel.sheet_names):
                    break

                pid = current_pid
                current_pid += 1

                left_sheet = excel.sheet_names[idx]
                right_sheet = excel.sheet_names[idx + 1]

                try:
                    df_left = pd.read_excel(excel, sheet_name=left_sheet, header=None)
                    df_right = pd.read_excel(excel, sheet_name=right_sheet, header=None)

                    # 提取特征 (位置与 ACL 相同)
                    # 左腿: 行 9-609, 列 2-8 (特征), 44-50 (归一化)
                    # 右腿: 行 9-609, 列 23-29 (特征), 65-71 (归一化)
                    feat_l = self._extract_leg_data(df_left, 9, 609, 2, 8)
                    norm_l = self._extract_leg_data(df_left, 9, 109, 44, 50)
                    feat_r = self._extract_leg_data(df_right, 9, 609, 23, 29)
                    norm_r = self._extract_leg_data(df_right, 9, 109, 65, 71)

                    # 添加记录
                    # 标签统一标记为 "关节炎"
                    data.append({
                        'person_id': pid, 'leg': 'left',
                        'features': feat_l, 'norm_features': norm_l,
                        'ACL_label': '关节炎', 'injure_label': '关节炎'
                    })
                    data.append({
                        'person_id': pid, 'leg': 'right',
                        'features': feat_r, 'norm_features': norm_r,
                        'ACL_label': '关节炎', 'injure_label': '关节炎'
                    })
                except Exception as e:
                    print(f"  ❌ 处理 ID {pid} ({left_sheet}) 时出错: {e}")
                    continue

        return pd.DataFrame(data)

    def _determine_labels(self, acl_side, injure_status):
        """根据患侧和半月板状态返回左右腿的具体标签"""
        # 默认为健康
        res = {
            'L_ACL': '健康', 'L_Injure': '健康',
            'R_ACL': '健康', 'R_Injure': '健康'
        }
        
        is_injure = (injure_status == 'injure')
        is_no_rec = (injure_status == 'no_record')
        
        meniscus_bad = '有合并半月板损伤' if is_injure else ('无记录' if is_no_rec else '无合并半月板损伤')
        meniscus_good = '健康' if not is_no_rec else '无记录' # 注意：原代码逻辑如果是无记录，健侧也标记为无记录

        if acl_side == 'LB': # 左腿患病
            res['L_ACL'] = '韧带断裂'
            res['L_Injure'] = meniscus_bad
            res['R_Injure'] = meniscus_good # 原逻辑健侧
        elif acl_side == 'RB': # 右腿患病
            res['R_ACL'] = '韧带断裂'
            res['R_Injure'] = meniscus_bad
            res['L_Injure'] = meniscus_good
            
        return res