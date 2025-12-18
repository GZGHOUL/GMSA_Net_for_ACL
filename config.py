import os

# 路径配置
DATASET_ROOT = 'D:/python_file/knee_joint_ALD/gait_shapeformer_re/Dataset/origin_data'
DATASET_ROOT2 = 'D:/python_file/knee_joint_ALD/gait_shapeformer_re/Dataset/Gait3'
RAW_XLSX_PATHS = {
    'ACL214': os.path.join(DATASET_ROOT, 'ACL214.xlsx'),
    'ACL109': os.path.join(DATASET_ROOT, 'ACL109.xlsx'),
    'Health333': os.path.join(DATASET_ROOT, 'Health333.xlsx'),
    'OA72': os.path.join(DATASET_ROOT, 'OA72.xlsx'),
    'OA60': os.path.join(DATASET_ROOT, 'OA60.xlsx')
}
LABEL_CSV_PATH = os.path.join(DATASET_ROOT, 'ACL_label.csv')

# 中间文件路径
INTERMEDIATE_HEALTH = os.path.join(DATASET_ROOT2, 'gait_health_dataset.csv')
INTERMEDIATE_PATIENT = os.path.join(DATASET_ROOT2, 'gait_patient_dataset.csv')
INTERMEDIATE_OA = os.path.join(DATASET_ROOT2, 'gait_oa_dataset.csv')

PROCESSED_HEALTH = os.path.join(DATASET_ROOT2, 'gait_health333_dataset_filtered.csv')
PROCESSED_PATIENT = os.path.join(DATASET_ROOT2, 'gait_patient_dataset_filtered_no_record.csv')
PROCESSED_OA = os.path.join(DATASET_ROOT2, 'gait_oa_dataset_filtered.csv')

# 输出文件路径
OUTPUT_TRAIN_A = os.path.join(DATASET_ROOT2, 'gait_train_A.csv')
OUTPUT_TRAIN_B = os.path.join(DATASET_ROOT2, 'gait_train_B.csv')
OUTPUT_TEST_A = os.path.join(DATASET_ROOT2, 'gait_test_A.csv')
OUTPUT_TEST_B = os.path.join(DATASET_ROOT2, 'gait_test_B.csv')

# 信号处理参数
FS = 60  # 采样频率
TARGET_LEN = 600
FILTER_PARAMS = {
    'notch_freq': 10,
    'width': 2,
    'ripple_db': 60,
    'cutoff': 6,
    'order': 6
}