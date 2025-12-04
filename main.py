import argparse
import yaml
from pathlib import Path

from src.data.simulate import simulate_social_platform
from src.data.dataset import build_dataframe
from src.data.loader_uci import load_uci_phishing, load_uci_sms_spam
from src.data.loader_twitter_accounts import load_twitter_accounts
from src.features.extract import FeatureExtractor
from src.signals.social_graph import build_social_graph, compute_social_signals
from src.models.detector import FraudDetector
from src.intervention.strategy import InterventionSimulator
from src.evaluation.evaluate import Evaluator


def load_config(config_path: str) -> dict:
	with open(config_path, 'r', encoding='utf-8') as f: #打开指定路径（config_path）的配置文件
		return yaml.safe_load(f) #使用 yaml 库安全地加载文件内容，并将其解析为 Python 字典（dict）后返回


def ensure_dirs():
	Path('outputs').mkdir(exist_ok=True)
	Path('configs').mkdir(exist_ok=True)
	Path('src').mkdir(exist_ok=True)
	Path('src/data').mkdir(parents=True, exist_ok=True)
	Path('src/features').mkdir(parents=True, exist_ok=True)
	Path('src/models').mkdir(parents=True, exist_ok=True)
	Path('src/signals').mkdir(parents=True, exist_ok=True)
	Path('src/intervention').mkdir(parents=True, exist_ok=True)
	Path('src/evaluation').mkdir(parents=True, exist_ok=True)


def main():
	parser = argparse.ArgumentParser() #创建一个参数解析器
	parser.add_argument('--config', type=str, default='configs/base.yaml')
	parser.add_argument('--dataset', type=str, default='simulated', choices=['simulated', 'uci_phishing', 'sms_spam', 'twitter_accounts'])
	parser.add_argument('--uci_dir', type=str, default='.')
	parser.add_argument('--twitter_dir', type=str, default='.')
	parser.add_argument('--twitter_subset', type=str, default=None, choices=[None, 'verified', 'botwiki'])
	args = parser.parse_args() #解析用户在命令行中输入的参数，并存入 args 对象中

	ensure_dirs()
	if not Path(args.config).exists():
		# 默认配置
		default_cfg = {
			"simulation": {"num_users": 2000, "fraud_ratio": 0.08, "seed": 42},
			"features": {"ngram_range": [1, 2], "max_features": 5000},
			"model": {"type": "xgb_like", "params": {"n_estimators": 300, "learning_rate": 0.08}},
			"intervention": {"threshold": 0.7, "cooldown_days": 7, "max_daily_flags": 100},
			"evaluation": {"test_size": 0.25, "random_state": 42}
		}
		Path(args.config).write_text(yaml.safe_dump(default_cfg, allow_unicode=True), encoding='utf-8')

	cfg = load_config(args.config)

	# 1) 读取数据
	if args.dataset == 'simulated':
		sim_data = simulate_social_platform(**cfg["simulation"])
		ds_name = 'simulated'
	elif args.dataset == 'uci_phishing':
		sim_data = load_uci_phishing(args.uci_dir)
		ds_name = 'uci_phishing'
	elif args.dataset == 'sms_spam':
		sim_data = load_uci_sms_spam(args.uci_dir)
		ds_name = 'sms_spam'
	elif args.dataset == 'twitter_accounts':
		sim_data = load_twitter_accounts(args.twitter_dir, subset=args.twitter_subset)
		ds_name = f"twitter_{args.twitter_subset or 'all'}"
	else:
		raise ValueError('unknown dataset')

	# 2) 构建整合数据集
	df_users, df_contents, df_interactions, df_labels = build_dataframe(sim_data)

	# 3) 构建社会图并计算社会信号
	G = build_social_graph(df_users, df_interactions)
	social_signals = compute_social_signals(G, df_users, df_contents, df_interactions)

	# 4) 特征工程（融合文本/用户画像/网络/时间/社会信号）
	extractor = FeatureExtractor(cfg["features"]) #初始化特征提取器
	X, y, meta = extractor.transform(
		df_users=df_users,
		df_contents=df_contents,
		df_interactions=df_interactions,
		df_labels=df_labels,
		social_signals=social_signals,
	)

	# 5) 训练检测模型
	detector = FraudDetector(cfg["model"]) #初始化欺诈检测器，传入模型配置
	clf, metrics, (test_idx, y_test, y_pred_proba) = detector.fit_and_eval(X, y, cfg["evaluation"])
	#调用此方法，该方法内部会划分训练集/测试集,在训练集上训练模型 clf，并在测试集上进行评估
	#返回clf:训练好的分类器对象,metrics:包含评估指标,(test_idx, y_test, y_pred_proba): 测试集的索引、真实标签和模型预测的概率

	# 6) 干预策略仿真（对齐测试集样本）
	meta_test = meta.iloc[test_idx].reset_index(drop=True)
	simulator = InterventionSimulator(cfg["intervention"])
	intervention_report = simulator.simulate(meta_test, y_pred_proba, df_interactions)

	# 7) 评估与可视化
	evaluator = Evaluator()
	save_path = f'outputs/metrics_{ds_name}.png'
	evaluator.plot_roc_pr(metrics, save_path=save_path)
	evaluator.summarize(metrics, intervention_report)

	print(f"流程完成。指标图已输出至 {save_path}")


if __name__ == '__main__':
	main()
