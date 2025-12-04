from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict


def _scan_source_dirs(root: str, subset: str | None) -> List[Tuple[str, int]]:
	"""Return list of (subdir_path, label) where label=1 for botwiki, 0 for verified.
	If subset is 'verified' or 'botwiki', filter accordingly.
	"""
	pairs: List[Tuple[str, int]] = []
	for name in os.listdir(root):
		low = name.lower()
		if os.path.isdir(os.path.join(root, name)) and ("verified-2019" in low or "botwiki-2019" in low):
			label = 1 if "bot" in low else 0
			if subset == 'verified' and label == 1:
				continue
			if subset == 'botwiki' and label == 0:
				continue
			pairs.append((os.path.join(root, name), label))
	return pairs


def _read_accounts_from_tsv(tsv_path: str) -> pd.DataFrame:
	try:
		df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
	except Exception:
		# sometimes ; separated
		df = pd.read_csv(tsv_path, sep=';', low_memory=False)
	return df


def _extract_user_columns(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series | None]:
	uid_col = None
	sn_col = None
	verified_col = None
	for c in df.columns:
		lc = str(c).lower()
		if lc in ["user_id", "user_id_str", "id_str", "id"]:
			uid_col = c
		if lc in ["screen_name", "username", "name"]:
			sn_col = c
		if lc in ["verified", "is_verified"]:
			verified_col = c
	if uid_col is None:
		uids = df.iloc[:, 0].astype(str)
	else:
		uids = df[uid_col].astype(str)
	snames = df[sn_col].astype(str) if sn_col is not None else pd.Series(["user"] * len(uids))
	ver_flags = df[verified_col] if verified_col is not None else None
	return uids, snames, ver_flags


def _read_tweets_json(json_path: str) -> List[Dict[str, Any]]:
	items: List[Dict[str, Any]] = []
	try:
		# Try JSON lines first
		with open(json_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					items.append(json.loads(line))
				except json.JSONDecodeError:
					# maybe whole file is a JSON array
					f.seek(0)
					items = json.load(f)
					break
	except Exception:
		pass
	return items


def load_twitter_accounts(directory: str, subset: str | None = None) -> Dict[str, Any]:
	"""
	支持目录结构：<dir>/verified-2019 与 <dir>/botwiki-2019，内部包含若干 .tsv 和 *_tweets.json。
	- subset: None（合并）、'verified'（仅verified-2019）、'botwiki'（仅botwiki-2019）。
	- 标签从文件夹名推断：botwiki-* 为 is_fraud=1，verified-* 为 0。
	- 用户ID与screen_name来自 tsv；若缺失，则从 tweets json 的 user 字段提取。
	- 为每个账号生成一条代表性文本（优先 tweets 文本，其次合成简介）。
	- 交互为空表（若后续提供关注/转发关系，可扩展）。
	"""
	sources = _scan_source_dirs(directory, subset)
	if not sources:
		# 兼容直接把文件放在 directory 的情况
		label_guess = 1 if (subset == 'botwiki' or "bot" in directory.lower()) else 0
		sources = [(directory, label_guess)]

	user_rows: List[Dict[str, Any]] = []
	user_to_texts: DefaultDict[str, List[str]] = defaultdict(list)
	uid_to_verified_hint: Dict[str, int] = {}

	for subdir, label in sources:
		# TSV user lists
		for fname in os.listdir(subdir):
			path = os.path.join(subdir, fname)
			low = fname.lower()
			if low.endswith('.tsv'):
				df = _read_accounts_from_tsv(path)
				uids, snames, ver_flags = _extract_user_columns(df)
				for idx, (u, sn) in enumerate(zip(uids, snames)):
					uid = f"tw_{u}"
					user_rows.append({"user_id": uid, "screen_name": sn, "is_fraud": label})
					if ver_flags is not None:
						try:
							uid_to_verified_hint[uid] = int(bool(ver_flags.iloc[idx]))
						except Exception:
							pass
			elif low.endswith('.json') and low.endswith('_tweets.json'):
				items = _read_tweets_json(path)
				for it in items:
					u = None
					sn = None
					tx = None
					if isinstance(it, dict):
						user = it.get('user') or {}
						u = str(user.get('id_str') or user.get('id') or '')
						sn = user.get('screen_name') or user.get('name') or 'user'
						tx = it.get('text') or it.get('full_text') or ''
					if u:
						uid = f"tw_{u}"
						user_rows.append({"user_id": uid, "screen_name": sn, "is_fraud": label})
						if tx:
							user_to_texts[uid].append(str(tx))

	# 去重
	acc = pd.DataFrame(user_rows).drop_duplicates("user_id")
	n = len(acc)
	if n == 0:
		raise ValueError('未能解析任何账号，请检查 tsv 列名或 tweets.json 结构')

	# 用户画像（避免用标签派生，防止特征泄漏）
	verified_vals = []
	for uid in acc['user_id']:
		if uid in uid_to_verified_hint:
			verified_vals.append(int(uid_to_verified_hint[uid]))
		else:
			verified_vals.append(int(np.random.binomial(1, 0.1)))
	users = pd.DataFrame({
		'user_id': acc['user_id'].tolist(),
		'account_age_days': np.random.randint(60, 3000, size=n),
		'followers': np.random.poisson(200, size=n),
		'following': np.random.poisson(300, size=n),
		'verified': np.array(verified_vals, dtype=int),
		'bot_score': np.random.beta(2, 8, size=n),
	})

	# 内容：优先选取 tweets 文本，没有则用简介占位
	base_time = datetime(2024, 3, 1)
	post_ids = []
	authors = []
	texts = []
	for i, row in acc.iterrows():
		uid = row['user_id']
		post_ids.append(f"p_tw_{i}")
		authors.append(uid)
		if user_to_texts.get(uid):
			texts.append(user_to_texts[uid][0])
		else:
			texts.append(f"profile intro from {row['screen_name']}")

	contents = pd.DataFrame({
		'post_id': post_ids,
		'author_id': authors,
		'text': texts,
		'created_at': [base_time + timedelta(minutes=i) for i in range(n)],
	})

	labels = pd.DataFrame({
		'post_id': post_ids,
		'is_fraud': acc['is_fraud'].astype(int).values,
	})

	interactions = pd.DataFrame(columns=['src_user', 'dst_user', 'post_id', 'time', 'type'])

	return {
		'users': users,
		'contents': contents,
		'interactions': interactions,
		'labels': labels,
	}
