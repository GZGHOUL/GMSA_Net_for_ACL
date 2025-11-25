import pandas as pd
import numpy as np

df_patient = pd.read_csv('Dataset/Gait2/gait_patient_dataset_filtered_no_record.csv')
df_health = pd.read_csv('Dataset/Gait2/gait_health333_dataset_filtered.csv')

# 获取所有person_id
health_ids = df_health['person_id'].unique()
patient_ids = df_patient['person_id'].unique()

# 随机打乱
np.random.seed(42)  # 保证可复现
health_ids_shuffled = np.random.permutation(health_ids)
patient_ids_shuffled = np.random.permutation(patient_ids)

# 70%训练，30%测试
n_health_train = int(len(health_ids) * 0.7)
n_patient_train = int(len(patient_ids) * 0.7)

health_train_ids = health_ids_shuffled[:n_health_train]
health_test_ids = health_ids_shuffled[n_health_train:]

patient_train_ids = patient_ids_shuffled[:n_patient_train]
patient_test_ids = patient_ids_shuffled[n_patient_train:]

# 按person_id筛选样本
df_train = pd.concat([
    df_health[df_health['person_id'].isin(health_train_ids)],
    df_patient[df_patient['person_id'].isin(patient_train_ids)]
], ignore_index=True)

df_test = pd.concat([
    df_health[df_health['person_id'].isin(health_test_ids)],
    df_patient[df_patient['person_id'].isin(patient_test_ids)]
], ignore_index=True)

def renum_person_id(df):
    new_ids = (np.arange(len(df)) // 2) + 1
    df['person_id'] = new_ids.astype(int)
    return df

df_train = renum_person_id(df_train)
df_test = renum_person_id(df_test)

def split_dataset(df):
    df_A = df[['person_id','leg', 'features', 'ACL_label', 'injure_label']]
    df_B = df[['person_id','leg', 'norm_features', 'ACL_label', 'injure_label']]
    return df_A, df_B

df_train_A, df_train_B = split_dataset(df_train)
df_test_A, df_test_B = split_dataset(df_test)

df_train_A.to_csv('Dataset/Gait2/gait_train_A.csv', index=False)
df_train_B.to_csv('Dataset/Gait2/gait_train_B.csv', index=False)
df_test_A.to_csv('Dataset/Gait2/gait_test_A.csv', index=False)
df_test_B.to_csv('Dataset/Gait2/gait_test_B.csv', index=False)

