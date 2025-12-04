from __future__ import annotations
import pandas as pd
import networkx as nx


def build_social_graph(df_users: pd.DataFrame, df_interactions: pd.DataFrame) -> nx.DiGraph:
	G = nx.DiGraph()
	for _, row in df_users.iterrows():
		G.add_node(row['user_id'])
	if not df_interactions.empty:
		# 以交互构边：src -> dst
		edges = df_interactions[['src_user', 'dst_user']].dropna()
		for _, r in edges.iterrows():
			G.add_edge(r['src_user'], r['dst_user'])
	return G


def compute_social_signals(G: nx.DiGraph, df_users: pd.DataFrame, df_contents: pd.DataFrame, df_interactions: pd.DataFrame) -> pd.DataFrame:
	if len(G) == 0:
		return pd.DataFrame({'author_id': [], 'deg': [], 'pr': [], 'clust': []})

	deg = dict(G.degree())
	try:
		pr = nx.pagerank(G, alpha=0.85)
	except Exception:
		pr = {n: 0.0 for n in G.nodes}
	try:
		clust_u = nx.clustering(G.to_undirected())
	except Exception:
		clust_u = {n: 0.0 for n in G.nodes}

	df = pd.DataFrame({
		'author_id': list(G.nodes),
		'deg': [deg.get(n, 0.0) for n in G.nodes],
		'pr': [pr.get(n, 0.0) for n in G.nodes],
		'clust': [clust_u.get(n, 0.0) for n in G.nodes],
	})
	return df
