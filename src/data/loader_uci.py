from __future__ import annotations
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any


def load_uci_phishing(base_dir: str) -> Dict[str, Any]:
	"""
	期望文件：
	- Phishing Websites Data Set: 常见文件名 'phishing.csv' 或 'PhishingData.csv' 或包含 30/31 个特征列与 'Result' 标签
	映射策略：
	- 每条样本→一条 content；text 由离散特征拼接为词
	- 无真实用户/关系：生成最小用户表（每条样本一个作者），交互留空
	- 标签：Result（1/-1、phishing/legitimate）→ is_fraud(1 表示诈骗)
	"""
	# 尝试若干常见文件名
	candidates = [
		'phishing.csv', 'PhishingData.csv', 'PhishingWebsitesData.csv', 'data.csv', 'dataset.csv'
	]
	csv_path = None
	for name in candidates:
		p = os.path.join(base_dir, name)
		if os.path.exists(p):
			csv_path = p
			break
	if csv_path is None:
		# 回退：查找目录下首个 .csv
		for fname in os.listdir(base_dir):
			if fname.lower().endswith('.csv'):
				csv_path = os.path.join(base_dir, fname)
				break
	if csv_path is None:
		raise FileNotFoundError('未找到 UCI Phishing 的 CSV 文件，请将解压后的 csv 放在此目录。')

	# 兼容不同分隔符
	read_ok = False
	for sep in [',', ';']:
		try:
			df = pd.read_csv(csv_path, sep=sep)
			# 非常短或单列说明分隔不对
			if df.shape[1] <= 1:
				continue
			read_ok = True
			break
		except Exception:
			continue
	if not read_ok:
		# 最后再试默认
		df = pd.read_csv(csv_path)

	# 寻找标签列
	label_col = None
	possible_labels = ['result', 'label', 'class', 'is_phishing', 'target']
	for c in df.columns:
		if c.lower().strip() in possible_labels:
			label_col = c
			break
	if label_col is None:
		# 回退：假设最后一列为标签
		label_col = df.columns[-1]

	# 构造文本：将离散特征拼成 token
	feature_cols = [c for c in df.columns if c != label_col]
	texts = []
	for _, row in df[feature_cols].iterrows():
		parts = []
		for c in feature_cols:
			val = row[c]
			parts.append(f"{c}_{val}")
		texts.append(' '.join(map(str, parts)))

	# 规范化标签
	raw_y = df[label_col]
	if raw_y.dtype == object:
		lower = raw_y.astype(str).str.lower().str.strip()
		is_fraud = lower.isin(['1', 'phishing', 'phish', 'fraud', 'true', 'yes']).astype(int)
	else:
		# 常见取值：1/-1 或 1/0
		vals = raw_y.astype(float)
		# 将 -1 视为正类或负类需根据数据集定义；UCI 中通常 1 表示合法、-1 表示钓鱼
		# 这里将 {1->0, -1->1}；若你的 CSV 恰好相反，可在运行时再取反
		is_fraud = (vals < 0).astype(int)

	n = len(df)
	post_ids = [f"p_ph_{i}" for i in range(n)]
	user_ids = [f"u_ph_{i}" for i in range(n)]
	base_time = datetime(2024, 1, 1)
	created_at = [base_time + timedelta(minutes=i) for i in range(n)]

	users = pd.DataFrame({
		'user_id': user_ids,
		'account_age_days': np.random.randint(30, 2000, size=n),
		'followers': np.random.poisson(50, size=n),
		'following': np.random.poisson(80, size=n),
		'verified': np.random.binomial(1, 0.02, size=n),
		'bot_score': np.random.beta(2, 8, size=n),
	})
	contents = pd.DataFrame({
		'post_id': post_ids,
		'author_id': user_ids,
		'text': texts,
		'created_at': created_at,
	})
	interactions = pd.DataFrame(columns=['src_user', 'dst_user', 'post_id', 'time', 'type'])
	labels = pd.DataFrame({
		'post_id': post_ids,
		'is_fraud': is_fraud.values.astype(int),
	})
	return {
		'users': users,
		'contents': contents,
		'interactions': interactions,
		'labels': labels,
	}


def load_uci_sms_spam(base_dir: str) -> Dict[str, Any]:
	"""
	期望文件：UCI SMS Spam Collection，常见文件名 'SMSSpamCollection'（制表或制表+空格分隔）或 csv
	映射：
	- 每条短信→一条 content，作者为虚拟用户
	- 标签 'spam' 为 is_fraud=1，'ham' 为 0
	"""
	# 常见文件名
	candidates = ['SMSSpamCollection', 'sms_spam.csv', 'SMSSpamCollection.csv']
	data_path = None
	for name in candidates:
		p = os.path.join(base_dir, name)
		if os.path.exists(p):
			data_path = p
			break
	if data_path is None:
		for fname in os.listdir(base_dir):
			if 'sms' in fname.lower() and fname.lower().endswith(('.txt', '.csv')):
				data_path = os.path.join(base_dir, fname)
				break
	if data_path is None:
		raise FileNotFoundError('未找到 SMS Spam 数据文件，请将 SMSSpamCollection 或 csv 放在此目录。')

	if data_path.lower().endswith('.csv'):
		df = pd.read_csv(data_path)
		# 兼容列
		label_col = None
		text_col = None
		for c in df.columns:
			lc = c.lower()
			if lc in ['label', 'category', 'class']:
				label_col = c
			if lc in ['message', 'text', 'sms', 'content']:
				text_col = c
		if label_col is None or text_col is None:
			raise ValueError('CSV 需包含 label/message 文本列')
	else:
		# 原始 txt：以制表分隔，第一列 label，第二列 text
		df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'text'])
		label_col, text_col = 'label', 'text'

	n = len(df)
	post_ids = [f"p_sms_{i}" for i in range(n)]
	user_ids = [f"u_sms_{i}" for i in range(n)]
	base_time = datetime(2024, 2, 1)
	created_at = [base_time + timedelta(minutes=i) for i in range(n)]

	users = pd.DataFrame({
		'user_id': user_ids,
		'account_age_days': np.random.randint(30, 2000, size=n),
		'followers': np.random.poisson(20, size=n),
		'following': np.random.poisson(30, size=n),
		'verified': np.random.binomial(1, 0.01, size=n),
		'bot_score': np.random.beta(2, 8, size=n),
	})
	contents = pd.DataFrame({
		'post_id': post_ids,
		'author_id': user_ids,
		'text': df[text_col].astype(str).tolist(),
		'created_at': created_at,
	})
	interactions = pd.DataFrame(columns=['src_user', 'dst_user', 'post_id', 'time', 'type'])
	labels = pd.DataFrame({
		'post_id': post_ids,
		'is_fraud': df[label_col].astype(str).str.lower().isin(['spam', '1', 'fraud']).astype(int),
	})
	return {
		'users': users,
		'contents': contents,
		'interactions': interactions,
		'labels': labels,
	}
