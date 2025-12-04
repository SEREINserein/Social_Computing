from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any


class InterventionSimulator:
	def __init__(self, cfg: Dict[str, Any]):
		self.threshold = cfg.get('threshold', 0.7)
		self.cooldown_days = cfg.get('cooldown_days', 7)
		self.max_daily_flags = cfg.get('max_daily_flags', 100)

	def simulate(self, meta: pd.DataFrame, y_pred_proba: np.ndarray, df_interactions: pd.DataFrame) -> Dict[str, Any]:
		# 基于阈值排序，限制每日最大处置量，对高风险内容的作者施加冷却（减少扩散）
		df = meta.copy()
		df['proba'] = y_pred_proba
		df.sort_values('proba', ascending=False, inplace=True)
		df['flag'] = (df['proba'] >= self.threshold).astype(int)

		# 每日上限控制
		df['date'] = pd.to_datetime(df['created_at']).dt.date
		flagged = []
		for d, group in df.groupby('date'):
			g = group[group['flag'] == 1].head(self.max_daily_flags)
			flagged.append(g)
		flagged = pd.concat(flagged) if len(flagged) else df.head(0)

		flag_authors = set(flagged['author_id'].tolist())
		# 冷却：近cooldown_days内对被标记作者的传播互动减少（仅用于报告统计）
		if not df_interactions.empty:
			inter = df_interactions.copy()
			inter['date'] = pd.to_datetime(inter['time']).dt.date
			affected = inter[inter['dst_user'].isin(flag_authors)]
			reduced_spread = int(len(affected) * 0.5)  # 假设干预减少50%扩散
		else:
			reduced_spread = 0

		report = {
			'total_flagged': int(flagged.shape[0]),
			'unique_authors_flagged': int(len(flag_authors)),
			'estimated_spread_reduction': int(reduced_spread),
			'threshold': float(self.threshold),
			'max_daily_flags': int(self.max_daily_flags),
		}
		return report
