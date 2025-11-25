import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json, re, numpy as np
from math import nan as _nan

def parse_features_json(s: str) -> np.ndarray:
    # 将各种大小写的 nan/NaN/NAN 替换成 JSON 的 null
    s = re.sub(r'\bnan\b', 'null', str(s), flags=re.IGNORECASE)
    data = json.loads(s)                  # -> list/list of lists
    arr = np.array(data, dtype=float)
    # json 的 null 会变成 np.nan 自动兼容
    return arr

df = pd.read_csv('Dataset/Gait2/gait_train_A.csv')

for i, row in df.iterrows():
    features = row['features']
    features = parse_features_json(features)
    plt.plot(features)
    plt.show()