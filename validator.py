import numpy as np
import pandas as pd
from utils import parse_features_to_array

class DataValidator:
    def __init__(self, threshold_max=1000, threshold_dead=1e-6):
        """
        threshold_max: 允许的最大绝对值，超过此值视为物理异常
        threshold_dead: 判断信号是否为"死信号"(方差极小)的阈值
        """
        self.threshold_max = threshold_max
        self.threshold_dead = threshold_dead

    def validate(self, df, dataset_name="Dataset"):
        print(f"\n🔍 开始校验数据集: {dataset_name}")
        report = {
            'nan_count': 0,
            'inf_count': 0,
            'extreme_count': 0,
            'dead_signal_count': 0,
            'shape_mismatch': 0,
            'problematic_ids': set()
        }
        
        target_shape = (600, 6) # 假设标准形状

        for idx, row in df.iterrows():
            pid = row.get('person_id', f'Idx_{idx}')
            leg = row.get('leg', '?')
            
            # 1. 解析数据
            try:
                features = parse_features_to_array(row['features'])
            except:
                print(f"  [Error] 无法解析数据 ID: {pid}")
                continue

            # 2. 检查形状
            if features.shape != target_shape:
                report['shape_mismatch'] += 1
                report['problematic_ids'].add(pid)
                # print(f"  [Shape] ID {pid} {leg} 形状错误: {features.shape}")
                continue

            # 3. 检查 NaN / Inf
            if np.isnan(features).any():
                report['nan_count'] += 1
                report['problematic_ids'].add(pid)
                print(f"  [NaN] ID {pid} {leg} 包含 NaN")
            
            if np.isinf(features).any():
                report['inf_count'] += 1
                report['problematic_ids'].add(pid)
                print(f"  [Inf] ID {pid} {leg} 包含 Inf")

            # 4. 检查数值极值 (物理合理性)
            # 步态数据如果是角度或位移，通常不会特别巨大
            max_val = np.max(np.abs(features))
            if max_val > self.threshold_max:
                report['extreme_count'] += 1
                report['problematic_ids'].add(pid)
                print(f"  [Extreme] ID {pid} {leg} 数值异常大: {max_val:.2f}")

            # 5. 检查死信号 (方差为0或极小)
            # 如果某一列全是同一个数（例如补0导致），方差接近0
            std_val = np.std(features, axis=0)
            if np.any(std_val < self.threshold_dead):
                report['dead_signal_count'] += 1
                # 这里不一定要报错，但值得警告
                # print(f"  [Dead] ID {pid} {leg} 包含死信号(方差≈0)")

        # 打印总结
        print("-" * 30)
        print(f"校验总结 ({len(df)} 样本):")
        print(f"  ❌ 形状错误: {report['shape_mismatch']}")
        print(f"  ❌ 含 NaN: {report['nan_count']}")
        print(f"  ❌ 含 Inf: {report['inf_count']}")
        print(f"  ⚠️ 数值过大 (> {self.threshold_max}): {report['extreme_count']}")
        print(f"  ⚠️ 死信号 (std < {self.threshold_dead}): {report['dead_signal_count']}")
        
        if len(report['problematic_ids']) > 0:
            print(f"  🚫 建议检查或剔除的 ID: {list(report['problematic_ids'])[:10]} ...")
        else:
            print("  ✅ 数据集看起来很健康！")
        print("-" * 30)
        
        return report