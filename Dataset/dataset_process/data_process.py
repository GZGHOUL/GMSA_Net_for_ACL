import numpy as np
import pandas as pd
from scipy.signal import kaiserord, firwin, filtfilt, butter
import json
import re

def check_gait_shape(csv_path):
    df = pd.read_csv(csv_path)
    error_person_ids = set()
    error_details = []
    for idx, row in df.iterrows():
        person_id = row['person_id']
        leg = row['leg']
        try:
            features = parse_features(row['features'])
        except Exception as e:
            error_person_ids.add(person_id)
            error_details.append((person_id, leg, f"parse error: {e}"))
            continue
        if features.shape != (600, 6):
            error_person_ids.add(person_id)
            error_details.append((person_id, leg, features.shape))
    print(f"文件 {csv_path} 中，步态数据shape不是(600,6)的有 {len(error_person_ids)} 人")
    print("详细信息（person_id, leg, shape/错误）：")
    for detail in error_details:
        print(detail)
    return error_person_ids, error_details

def parse_features(s: str) -> np.ndarray:
    """Parse features cell string to numpy array, robust to 'nan'/NaN tokens.

    Strategy:
    - Replace bare nan/NaN with JSON null, then json.loads and cast to float.
    - Fallback: restricted eval allowing {'nan': float('nan'), 'NaN': float('nan')}.
    """
    if not isinstance(s, str):
        return np.asarray(s, dtype=float)
    # Normalize whitespace
    text = s.strip()
    # Replace bare nan variants with JSON null
    text_json = re.sub(r"\b(?:nan|NaN|NAN)\b", "null", text)
    try:
        return np.array(json.loads(text_json), dtype=float)
    except Exception:
        # Controlled eval fallback (no builtins, only nan symbols permitted)
        try:
            value = eval(text, {"__builtins__": None}, {"nan": float('nan'), "NaN": float('nan')})
            return np.array(value, dtype=float)
        except Exception as e:
            raise e

error_ids1, details1 = check_gait_shape('Dataset/Gait2/gait_health333_dataset.csv')

df = pd.read_csv('Dataset/Gait2/gait_health333_dataset.csv')
fixed_count = 0

for idx, row in df.iterrows():
    features = parse_features(row['features'])
    if features.shape == (599, 6):
        # 补一行全0
        zero_row = np.zeros((1, 6))
        features = np.vstack([features, zero_row])
        df.at[idx, 'features'] = features.tolist()
        fixed_count += 1
    elif features.shape == (598, 6):
        # 补两行全0
        zero_row = np.zeros((2, 6))
        features = np.vstack([features, zero_row])
        df.at[idx, 'features'] = features.tolist()
        fixed_count += 1
    elif features.shape == (597, 6):
        # 补两行全0
        zero_row = np.zeros((3, 6))
        features = np.vstack([features, zero_row])
        df.at[idx, 'features'] = features.tolist()
        fixed_count += 1
print(f"已修正 {fixed_count} 条 shape 不为 (600,6) 的数据。")



# 假设采样频率
fs = 60  # 例如100Hz，请根据实际情况修改

def notch_filter_kaiser(data, fs, notch_freq=10, width=2, ripple_db=60):
    nyq = fs / 2
    # 陷波频率范围
    low = notch_freq - width/2
    high = notch_freq + width/2
    # 保证频率范围合法
    if low <= 0:
        low = 0.01
    if high >= nyq:
        high = nyq - 0.01
    freq = [low / nyq, high / nyq]
    # 计算阶数和beta
    N, beta = kaiserord(ripple_db, width / nyq)
    if N % 2 == 0:
        N += 1
    taps = firwin(N, freq, window=('kaiser', beta), pass_zero='bandstop')
    return filtfilt(taps, 1.0, data, axis=0)

def butter_lowpass_filter(data, fs, cutoff=6, order=6):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data, axis=0)

def process_features(features, fs):
    filtered = notch_filter_kaiser(features, fs, notch_freq=10, width=2, ripple_db=60)
    filtered = butter_lowpass_filter(filtered, fs, cutoff=6, order=6)
    return filtered

# 对DataFrame批量处理
# df = pd.read_csv('Dataset/Gait2/gait_patient_dataset.csv')
for i, row in df.iterrows():
    features = parse_features(row['features'])  # 还原为numpy数组
    filtered_features = process_features(features, fs)
    df.at[i, 'features'] = filtered_features.tolist()

df.to_csv('Dataset/Gait2/gait_health333_dataset_filtered.csv', index=False)

error_ids1, details1 = check_gait_shape('Dataset/Gait2/gait_health333_dataset_filtered.csv')