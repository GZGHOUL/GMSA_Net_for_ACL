import pandas as pd
import os
from config import *
from builder import DatasetBuilder
from processor import DataCleaner, SignalProcessor
from splitter import DatasetSplitter
from validator import DataValidator

def main():
    print("=== 开始数据处理流程 ===")
    
    # 1. 构建数据集 (Build)
    # ---------------------------------------------------------
    builder = DatasetBuilder(RAW_XLSX_PATHS, LABEL_CSV_PATH)
    
    # 构建或读取健康人数据
    if not os.path.exists(INTERMEDIATE_HEALTH):
        df_health = builder.build_health_dataset()
        df_health.to_csv(INTERMEDIATE_HEALTH, index=False)
    else:
        print("加载现有健康人中间文件...")
        df_health = pd.read_csv(INTERMEDIATE_HEALTH)

    # 构建或读取患者数据
    if not os.path.exists(INTERMEDIATE_PATIENT):
        df_patient = builder.build_patient_dataset()
        df_patient.to_csv(INTERMEDIATE_PATIENT, index=False)
    else:
        print("加载现有患者中间文件...")
        df_patient = pd.read_csv(INTERMEDIATE_PATIENT)

    if not os.path.exists(INTERMEDIATE_OA):
        df_oa = builder.build_oa_dataset()
        df_oa.to_csv(INTERMEDIATE_OA, index=False)
    else:
        print("加载现有OA患者中间文件...")
        df_oa = pd.read_csv(INTERMEDIATE_OA)

    # 2. 数据清洗与处理 (Clean & Process)
    # ---------------------------------------------------------
    processor = SignalProcessor(fs=FS, params=FILTER_PARAMS)
    cleaner = DataCleaner()

    # 初始化校验器
    validator = DataValidator(threshold_max=2000)

    # --- 处理健康人数据 ---
    print("\n--- 处理健康人数据 ---")
    # 1. 修正形状 & 滤波
    df_health = cleaner.fix_shape_and_filter(df_health, processor)
    # 2. [新增] 剔除坏数据 (那5个NaN样本将被删掉)
    df_health = cleaner.drop_invalid_samples(df_health)
    # 3. 最终校验
    validator.validate(df_health, "健康人数据集") 
    
    df_health.to_csv(PROCESSED_HEALTH, index=False)

    # --- 处理患者数据 ---
    print("\n--- 处理患者数据 ---")
    df_patient = cleaner.remove_no_record(df_patient)
    df_patient = cleaner.fix_shape_and_filter(df_patient, processor)
    # 2. [新增] 剔除坏数据 (虽然目前看起来是健康的，防患于未然)
    df_patient = cleaner.drop_invalid_samples(df_patient)
    # 3. 最终校验
    validator.validate(df_patient, "患者数据集")
    
    df_patient.to_csv(PROCESSED_PATIENT, index=False)

    # --- [新增] 处理 OA 患者 ---
    print("\n--- 处理 OA 患者数据 ---")
    # OA 数据没有 "无记录" 的情况，直接处理
    df_oa = cleaner.fix_shape_and_filter(df_oa, processor)
    df_oa = cleaner.drop_invalid_samples(df_oa)
    validator.validate(df_oa, "OA患者数据集")
    df_oa.to_csv(PROCESSED_OA, index=False)

    # 3. 数据集划分 (Split)
    # ---------------------------------------------------------
    print("\n--- 划分训练集与测试集 ---")

    df_all_patients = pd.concat([df_patient, df_oa], ignore_index=True)

    print(f"合并后患者总数: {len(df_all_patients)} (含 ACL + OA)")

    splitter = DatasetSplitter(seed=42, train_ratio=0.7)
    
    output_paths = {
        'train_A': OUTPUT_TRAIN_A,
        'train_B': OUTPUT_TRAIN_B,
        'test_A': OUTPUT_TEST_A,
        'test_B': OUTPUT_TEST_B
    }
    
    splitter.split_and_save(df_health, df_all_patients, output_paths)
    
    print("\n=== 全部流程执行完毕 ===")

if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(DATASET_ROOT2, exist_ok=True)
    main()