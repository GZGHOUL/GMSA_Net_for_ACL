import pandas as pd
import numpy as np


df = pd.read_csv('Dataset/Gait2/gait_patient_dataset_filtered.csv')
print(df)
for i, row in df.iterrows():
    injure_label = row['injure_label']
    if injure_label == '无记录':
        df = df.drop(i)

new_ids = (np.arange(len(df)) // 2) + 1
df['person_id'] = new_ids.astype(int)
print(df)

df.to_csv('Dataset/Gait2/gait_patient_dataset_filtered_no_record.csv', index=False)