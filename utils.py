import json
import re
import numpy as np
import pandas as pd

def parse_features_to_array(s: str) -> np.ndarray:
    """
    将 CSV 中的字符串特征转换为 NumPy 数组。
    处理 'nan', 'NaN' 等特殊情况。
    """
    if not isinstance(s, str):
        return np.asarray(s, dtype=float)
    
    text = s.strip()
    # 替换 bare nan 为 JSON null
    text_json = re.sub(r"\b(?:nan|NaN|NAN)\b", "null", text)
    
    try:
        return np.array(json.loads(text_json), dtype=float)
    except Exception:
        try:
            # 备用方案
            value = eval(text, {"__builtins__": None}, {"nan": float('nan'), "NaN": float('nan')})
            return np.array(value, dtype=float)
        except Exception as e:
            raise ValueError(f"解析特征出错: {e}")

def read_csv_with_fallback(file_path):
    """尝试多种编码读取CSV"""
    encoding_candidates = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp936", "latin1"]
    for enc in encoding_candidates:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {file_path}")