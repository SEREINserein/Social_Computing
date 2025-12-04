from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
	def __init__(self, cfg: Dict[str, Any]):
		self.cfg = cfg
		self.vectorizer = TfidfVectorizer(
			norm='l2',
			ngram_range=tuple(cfg.get('ngram_range', [1, 2])),
			max_features=cfg.get('max_features', 5000),
			min_df=2,
		)

	def _text_features(self, df_contents: pd.DataFrame) -> sparse.csr_matrix:
		texts = df_contents['text'].fillna('').astype(str).tolist()
		return self.vectorizer.fit_transform(texts)

	def _user_features(self, df_users: pd.DataFrame, df_contents: pd.DataFrame) -> np.ndarray:
		uid_to_row = {u: i for i, u in enumerate(df_users['user_id'].tolist())}
		author_idx = df_contents['author_id'].map(uid_to_row).fillna(-1).astype(int).values
		u = df_users
		feat = np.stack([
			u['account_age_days'].values,
			u['followers'].values,
			u['following'].values,
			u['verified'].values,
			u['bot_score'].values,
		], axis=1)
		feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
		post_feat = feat[np.clip(author_idx, 0, len(feat)-1)]
		return post_feat

	def _temporal_features(self, df_contents: pd.DataFrame) -> np.ndarray:
		# 简单的时间特征：小时/星期几
		created = pd.to_datetime(df_contents['created_at'])
		hour = created.dt.hour.values.reshape(-1, 1)
		wday = created.dt.weekday.values.reshape(-1, 1)
		feat = np.concatenate([hour, wday], axis=1).astype(float)
		feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
		return feat

	def _social_signal_features(self, df_contents: pd.DataFrame, social_signals: pd.DataFrame) -> np.ndarray:
		merged = df_contents[['post_id', 'author_id']].merge(social_signals, on='author_id', how='left')
		cols = ['deg', 'pr', 'clust']
		for c in cols:
			if c not in merged:
				merged[c] = 0.0
		vals = merged[cols].fillna(0.0).values.astype(float)
		vals = (vals - vals.mean(0)) / (vals.std(0) + 1e-6)
		return vals

	def transform(
		self,
		df_users: pd.DataFrame,
		df_contents: pd.DataFrame,
		df_interactions: pd.DataFrame,
		df_labels: pd.DataFrame,
		social_signals: pd.DataFrame,
	) -> Tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame]:
		X_text = self._text_features(df_contents)
		X_user = self._user_features(df_users, df_contents)
		X_time = self._temporal_features(df_contents)
		X_sig = self._social_signal_features(df_contents, social_signals)

		X_dense = np.concatenate([X_user, X_time, X_sig], axis=1)
		X = sparse.hstack([X_text, sparse.csr_matrix(X_dense)], format='csr')

		label_map = df_labels.set_index('post_id')['is_fraud']
		y = df_contents['post_id'].map(label_map).fillna(0).astype(int).values

		meta = df_contents[['post_id', 'author_id', 'created_at']].copy()
		return X, y, meta
