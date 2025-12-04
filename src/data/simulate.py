from __future__ import annotations
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any


FRAUD_PHRASES = [
	"高额回报",
	"稳赚不赔",
	"官方认证理财",
	"加VX私聊",
	"限时保本计划",
	"内部消息",
]

NORMAL_PHRASES = [
	"旅游攻略",
	"美食分享",
	"学习笔记",
	"每日健身",
	"数码测评",
	"电影推荐",
]


def _random_datetime(start: datetime, end: datetime) -> datetime:
	span = end - start
	return start + timedelta(seconds=random.randint(0, int(span.total_seconds())))


def simulate_social_platform(num_users: int = 2000, fraud_ratio: float = 0.08, seed: int = 42) -> Dict[str, Any]:
	random.seed(seed)
	np.random.seed(seed)

	# 用户画像
	user_ids = [f"u_{i}" for i in range(num_users)]
	account_age_days = np.random.randint(7, 2000, size=num_users)
	followers = np.random.zipf(2, size=num_users)
	followers = np.clip(followers, 0, 50000)
	following = np.random.poisson(100, size=num_users)
	verified = np.random.binomial(1, 0.05, size=num_users)
	bot_score = np.clip(np.random.beta(2, 8, size=num_users) + verified * 0.05, 0, 1)

	users = pd.DataFrame({
		'user_id': user_ids,
		'account_age_days': account_age_days,
		'followers': followers,
		'following': following,
		'verified': verified,
		'bot_score': bot_score,
	})

	# 发帖（内容）
	num_posts = int(num_users * 1.2)
	post_ids = [f"p_{i}" for i in range(num_posts)]
	authors = np.random.choice(user_ids, size=num_posts, replace=True)
	is_fraud = np.random.binomial(1, fraud_ratio, size=num_posts)
	base_time = datetime(2024, 1, 1)

	texts = []
	for f in is_fraud:
		if f:
			phrases = np.random.choice(FRAUD_PHRASES, size=np.random.randint(1, 3), replace=False)
			noise = ''.join(np.random.choice(list('零售理财币股基盈保权重优惠'), size=np.random.randint(5, 15)))
			texts.append(" ".join(phrases) + " " + noise)
		else:
			phrases = np.random.choice(NORMAL_PHRASES, size=np.random.randint(1, 3), replace=False)
			noise = ''.join(np.random.choice(list('生活日常学习科技运动旅行'), size=np.random.randint(5, 15)))
			texts.append(" ".join(phrases) + " " + noise)

	created_at = [base_time + timedelta(minutes=int(i * np.random.gamma(1.5, 2))) for i in range(num_posts)]

	contents = pd.DataFrame({
		'post_id': post_ids,
		'author_id': authors,
		'text': texts,
		'created_at': created_at,
		'is_fraud': is_fraud,
	})

	# 交互：转发/评论/私信，构成有向关系与扩散
	interaction_types = ['retweet', 'reply', 'dm']
	rows = []
	for pid, author, t0 in zip(post_ids, authors, created_at):
		pop = np.random.poisson(3 if is_fraud[contents.index[contents['post_id'] == pid][0]] else 2)
		pop = int(pop + np.random.randint(0, 5))
		for _ in range(pop):
			v = np.random.choice(user_ids)
			if v == author:
				continue
			itime = t0 + timedelta(minutes=np.random.randint(1, 1440))
			rows.append({
				'src_user': v,
				'dst_user': author,
				'post_id': pid,
				'time': itime,
				'type': np.random.choice(interaction_types, p=[0.6, 0.3, 0.1]),
			})

	interactions = pd.DataFrame(rows)
	if not interactions.empty:
		interactions.sort_values('time', inplace=True)

	labels = contents[['post_id', 'is_fraud']].copy()

	return {
		'users': users,
		'contents': contents,
		'interactions': interactions,
		'labels': labels,
	}
