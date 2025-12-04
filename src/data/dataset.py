from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Tuple


def build_dataframe(sim_data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	users = sim_data['users'].copy()
	contents = sim_data['contents'].copy()
	interactions = sim_data['interactions'].copy()
	labels = sim_data['labels'].copy()

	# 基本清洗
	users.drop_duplicates('user_id', inplace=True)
	contents.drop_duplicates('post_id', inplace=True)
	if not interactions.empty:
		interactions.dropna(subset=['src_user', 'dst_user'], inplace=True)

	return users, contents, interactions, labels
