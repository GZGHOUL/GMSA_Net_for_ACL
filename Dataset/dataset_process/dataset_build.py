import pandas as pd

def _read_csv_with_fallback(file_path):
	"""Robustly read CSV with multiple encoding fallbacks for Windows/Excel exports."""
	encoding_candidates = [
		"utf-8",
		"utf-8-sig",
		"gb18030",
		"gbk",
		"cp936",
		"utf-16",
		"utf-16le",
		"utf-16be",
		"latin1",
	]
	last_error = None
	for enc in encoding_candidates:
		try:
			return pd.read_csv(file_path, encoding=enc)
		except UnicodeDecodeError as e:
			last_error = e
			continue
		except Exception as e:
			# Keep trying other encodings as some may raise UnicodeError subclasses
			last_error = e
			continue
	# If all attempts failed, raise the last error for visibility
	raise last_error

xlsx1_path = 'Dataset/Gait2/ACL214.xlsx'
xlsx2_path = 'Dataset/Gait2/ACL109.xlsx'
xlsx3_path = 'Dataset/Gait2/Health333.xlsx'
excel1_file = pd.ExcelFile(xlsx1_path, engine='openpyxl')
excel2_file = pd.ExcelFile(xlsx2_path, engine='openpyxl')
excel3_file = pd.ExcelFile(xlsx3_path, engine='openpyxl')
label_path = 'Dataset/Gait2/ACL_label.csv'
label_file = _read_csv_with_fallback(label_path)
# print(len(excel1_file.sheet_names))
# print(len(excel2_file.sheet_names))
# print(label_file.iloc[0])

data = []

# for idx in range(0, len(excel3_file.sheet_names), 2):
#     left_sheet = excel3_file.sheet_names[idx]
#     right_sheet = excel3_file.sheet_names[idx + 1]
#     person_id = idx // 2 + 1

#     # 读取左膝sheet
#     df_left = pd.read_excel(xlsx3_path, sheet_name=left_sheet, header=None, engine='openpyxl')
#     # 读取右膝sheet
#     df_right = pd.read_excel(xlsx3_path, sheet_name=right_sheet, header=None, engine='openpyxl')


#     # 左膝数据
#     features_left = df_left.iloc[9:609, 2:8].astype(float)
#     features_left = features_left.interpolate(method='linear', axis=0, limit_direction='both')
#     features_left = features_left.fillna(method='bfill').fillna(method='ffill')
#     features_left = features_left.values.tolist()

#     norm_features_left = df_left.iloc[9:109, 44:50].astype(float)
#     norm_features_left = norm_features_left.interpolate(method='linear', axis=0, limit_direction='both')
#     norm_features_left = norm_features_left.fillna(method='bfill').fillna(method='ffill')
#     norm_features_left = norm_features_left.values.tolist()

#     # 右膝数据
#     features_right = df_right.iloc[9:609, 23:29].astype(float)
#     features_right = features_right.interpolate(method='linear', axis=0, limit_direction='both')
#     features_right = features_right.fillna(method='bfill').fillna(method='ffill')
#     features_right = features_right.values.tolist()

#     norm_features_right = df_right.iloc[9:109, 65:71].astype(float)
#     norm_features_right = norm_features_right.interpolate(method='linear', axis=0, limit_direction='both')
#     norm_features_right = norm_features_right.fillna(method='bfill').fillna(method='ffill')
#     norm_features_right = norm_features_right.values.tolist()

#     # 按标签分配左右腿的最终标签
#     label_ACL_right = '健康'
#     label_injure_right = '健康'
#     label_ACL_left = '健康'
#     label_injure_left = '健康'


#     data.append({
#         'person_id': person_id,
#         'leg': 'left',
#         'features': features_left,
#         'norm_features': norm_features_left,
#         'ACL_label': label_ACL_left,
#         'injure_label': label_injure_left
#     })
#     data.append({
#         'person_id': person_id,
#         'leg': 'right',
#         'features': features_right,
#         'norm_features': norm_features_right,
#         'ACL_label': label_ACL_right,
#         'injure_label': label_injure_right
#     })

# print(data)
# df_out = pd.DataFrame(data)
# df_out.to_csv('Dataset/Gait2/gait_health_dataset.csv', index=False)


for idx in range(4, len(excel1_file.sheet_names), 2):
    left_sheet = excel1_file.sheet_names[idx]
    right_sheet = excel1_file.sheet_names[idx + 1]
    person_id = (idx-4) // 2 + 1

    # 读取左膝sheet
    df_left = pd.read_excel(xlsx1_path, sheet_name=left_sheet, header=None, engine='openpyxl')
    # 读取右膝sheet
    df_right = pd.read_excel(xlsx1_path, sheet_name=right_sheet, header=None, engine='openpyxl')

    label_value1 = df_left.iloc[1, 7]
    label_value2 = label_file.iloc[person_id - 1, 17]

    if label_value1 == '正常':
        ACL_label = 'RB'
    else:
        ACL_label = 'LB'
    
    if label_value2 == 0:
        injure_label = 'no_injure'
    elif label_value2 == 1:
        injure_label = 'injure'
    else:
        injure_label = 'no_record'

    # 左膝数据
    # features_left = df_left.iloc[9:109, 44:50].astype(float)
    features_left = df_left.iloc[9:609, 2:8].astype(float)
    features_left = features_left.interpolate(method='linear', axis=0, limit_direction='both')
    features_left = features_left.fillna(method='bfill').fillna(method='ffill')
    features_left = features_left.values.tolist()

    norm_features_left = df_left.iloc[9:109, 44:50].astype(float)
    norm_features_left = norm_features_left.interpolate(method='linear', axis=0, limit_direction='both')
    norm_features_left = norm_features_left.fillna(method='bfill').fillna(method='ffill')
    norm_features_left = norm_features_left.values.tolist()

    # 右膝数据
    # features_right = df_right.iloc[9:109, 65:71].astype(float)
    features_right = df_right.iloc[9:609, 23:29].astype(float)
    features_right = features_right.interpolate(method='linear', axis=0, limit_direction='both')
    features_right = features_right.fillna(method='bfill').fillna(method='ffill')
    features_right = features_right.values.tolist()

    norm_features_right = df_right.iloc[9:109, 65:71].astype(float)
    norm_features_right = norm_features_right.interpolate(method='linear', axis=0, limit_direction='both')
    norm_features_right = norm_features_right.fillna(method='bfill').fillna(method='ffill')
    norm_features_right = norm_features_right.values.tolist()

    # 按标签分配左右腿的最终标签
    if ACL_label == 'RB' and injure_label == 'no_injure':
        label_ACL_right = '韧带断裂'
        label_injure_right = '无合并半月板损伤'
        label_ACL_left = '健康'
        label_injure_left = '健康'
    elif ACL_label == 'RB' and injure_label == 'injure':
        label_ACL_right = '韧带断裂'
        label_injure_right = '有合并半月板损伤'
        label_ACL_left = '健康'
        label_injure_left = '健康'
    elif ACL_label == 'RB' and injure_label == 'no_record':
        label_ACL_right = '韧带断裂'
        label_injure_right = '无记录'
        label_ACL_left = '健康'
        label_injure_left = '无记录'
    elif ACL_label == 'LB' and injure_label == 'no_injure':
        label_ACL_right = '健康'
        label_injure_right = '健康'
        label_ACL_left = '韧带断裂'
        label_injure_left = '无合并半月板损伤'
    elif ACL_label == 'LB' and injure_label == 'injure':
        label_ACL_right = '健康'
        label_injure_right = '健康'
        label_ACL_left = '韧带断裂'
        label_injure_left = '有合并半月板损伤'
    elif ACL_label == 'LB' and injure_label == 'no_record':
        label_ACL_right = '健康'
        label_injure_right = '无记录'
        label_ACL_left = '韧带断裂'
        label_injure_left = '无记录'

    data.append({
        'person_id': person_id,
        'leg': 'left',
        'features': features_left,
        'norm_features': norm_features_left,
        'ACL_label': label_ACL_left,
        'injure_label': label_injure_left
    })
    data.append({
        'person_id': person_id,
        'leg': 'right',
        'features': features_right,
        'norm_features': norm_features_right,
        'ACL_label': label_ACL_right,
        'injure_label': label_injure_right
    })

for idx in range(0, len(excel2_file.sheet_names), 2):
    left_sheet = excel2_file.sheet_names[idx]
    right_sheet = excel2_file.sheet_names[idx + 1]
    person_id = idx // 2 + 1 + 212

    # 读取左膝sheet
    df_left = pd.read_excel(xlsx2_path, sheet_name=left_sheet, header=None, engine='openpyxl')
    # 读取右膝sheet
    df_right = pd.read_excel(xlsx2_path, sheet_name=right_sheet, header=None, engine='openpyxl')

    label_value1 = df_left.iloc[1, 7]
    label_value2 = label_file.iloc[person_id-1, 17]

    if label_value1 == '正常':
        ACL_label = 'RB'
    else:
        ACL_label = 'LB'
    
    if label_value2 == 0:
        injure_label = 'no_injure'
    elif label_value2 == 1:
        injure_label = 'injure'
    else:
        injure_label = 'no_record'

    # 左膝数据
    # features_left = df_left.iloc[9:109, 44:50].astype(float)
    features_left = df_left.iloc[9:609, 2:8].astype(float)
    features_left = features_left.interpolate(method='linear', axis=0, limit_direction='both')
    features_left = features_left.fillna(method='bfill').fillna(method='ffill')
    features_left = features_left.values.tolist()

    norm_features_left = df_left.iloc[9:109, 44:50].astype(float)
    norm_features_left = norm_features_left.interpolate(method='linear', axis=0, limit_direction='both')
    norm_features_left = norm_features_left.fillna(method='bfill').fillna(method='ffill')
    norm_features_left = norm_features_left.values.tolist()

    # 右膝数据
    # features_right = df_right.iloc[9:109, 65:71].astype(float)
    features_right = df_right.iloc[9:609, 23:29].astype(float)
    features_right = features_right.interpolate(method='linear', axis=0, limit_direction='both')
    features_right = features_right.fillna(method='bfill').fillna(method='ffill')
    features_right = features_right.values.tolist()

    norm_features_right = df_right.iloc[9:109, 65:71].astype(float)
    norm_features_right = norm_features_right.interpolate(method='linear', axis=0, limit_direction='both')
    norm_features_right = norm_features_right.fillna(method='bfill').fillna(method='ffill')
    norm_features_right = norm_features_right.values.tolist()

    # 按标签分配左右腿的最终标签
    if ACL_label == 'RB' and injure_label == 'no_injure':
        label_ACL_right = '韧带断裂'
        label_injure_right = '无合并半月板损伤'
        label_ACL_left = '健康'
        label_injure_left = '健康'
    elif ACL_label == 'RB' and injure_label == 'injure':
        label_ACL_right = '韧带断裂'
        label_injure_right = '有合并半月板损伤'
        label_ACL_left = '健康'
        label_injure_left = '健康'
    elif ACL_label == 'RB' and injure_label == 'no_record':
        label_ACL_right = '韧带断裂'
        label_injure_right = '无记录'
        label_ACL_left = '健康'
        label_injure_left = '无记录'
    elif ACL_label == 'LB' and injure_label == 'no_injure':
        label_ACL_right = '健康'
        label_injure_right = '健康'
        label_ACL_left = '韧带断裂'
        label_injure_left = '无合并半月板损伤'
    elif ACL_label == 'LB' and injure_label == 'injure':
        label_ACL_right = '健康'
        label_injure_right = '健康'
        label_ACL_left = '韧带断裂'
        label_injure_left = '有合并半月板损伤'
    elif ACL_label == 'LB' and injure_label == 'no_record':
        label_ACL_right = '健康'
        label_injure_right = '无记录'
        label_ACL_left = '韧带断裂'
        label_injure_left = '无记录'

    data.append({
        'person_id': person_id,
        'leg': 'left',
        'features': features_left,
        'norm_features': norm_features_left,
        'ACL_label': label_ACL_left,
        'injure_label': label_injure_left
    })
    data.append({
        'person_id': person_id,
        'leg': 'right',
        'features': features_right,
        'norm_features': norm_features_right,
        'ACL_label': label_ACL_right,
        'injure_label': label_injure_right
    })


df_out = pd.DataFrame(data)
df_out.to_csv('Dataset/Gait2/gait_patient_dataset.csv', index=False)