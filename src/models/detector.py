from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from scipy import sparse


class FraudDetector:
	def __init__(self, cfg: Dict[str, Any]):
		self.cfg = cfg
		self.model = None

	def fit_and_eval(self, X, y, eval_cfg: Dict[str, Any]):
		indices = np.arange(len(y))
		X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
			X, y, indices,
			test_size=eval_cfg.get('test_size', 0.25),
			random_state=eval_cfg.get('random_state', 42),
			stratify=None if len(np.unique(y)) == 1 else y
		)

		# 单一类别：切换无监督异常检测
		if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
			is_sparse = sparse.issparse(X_train)
			Xtr = X_train if not is_sparse else X_train
			Xte = X_test if not is_sparse else X_test
			oc = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
			oc.fit(Xtr)
			# 越异常分数越高，转为 [0,1]
			scores = -oc.score_samples(Xte)
			s_min, s_max = float(np.min(scores)), float(np.max(scores))
			proba = (scores - s_min) / (s_max - s_min + 1e-8)
			metrics = {
				'roc_auc': None,
				'pr_auc': None,
				'roc_curve': ([], []),
				'pr_curve': ([], []),
				'note': 'single_class_subset_used_isolation_forest',
			}
			self.model = oc
			return self.model, metrics, (test_idx, y_test, proba)

		is_sparse = sparse.issparse(X_train)
		params = self.cfg.get('params', {})

		if is_sparse:
			# 稀疏友好：LogisticRegression(saga)
			self.model = LogisticRegression(
				solver='saga',
				penalty='l2',
				C=1.0,
				max_iter=int(params.get('max_iter', 300)),
				n_jobs=-1,
				verbose=0,
			)
			self.model.fit(X_train, y_train)
			proba = self.model.predict_proba(X_test)[:, 1]
		else:
			# 稠密：保持原 HGB
			self.model = HistGradientBoostingClassifier(
				max_depth=None,
				learning_rate=params.get('learning_rate', 0.08),
				max_iter=params.get('n_estimators', 300),
				l2_regularization=1e-4,
				random_state=42,
			)
			X_train_arr = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
			X_test_arr = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
			self.model.fit(X_train_arr, y_train)
			proba = self.model.predict_proba(X_test_arr)[:, 1]

		roc_auc = roc_auc_score(y_test, proba)
		pr_auc = average_precision_score(y_test, proba)
		fpr, tpr, _ = roc_curve(y_test, proba)
		prec, rec, _ = precision_recall_curve(y_test, proba)
		metrics = {
			'roc_auc': float(roc_auc),
			'pr_auc': float(pr_auc),
			'roc_curve': (fpr.tolist(), tpr.tolist()),
			'pr_curve': (prec.tolist(), rec.tolist()),
		}
		return self.model, metrics, (test_idx, y_test, proba)
