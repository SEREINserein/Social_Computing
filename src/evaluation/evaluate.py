from __future__ import annotations
from typing import Dict, Any
import matplotlib.pyplot as plt


class Evaluator:
	def plot_roc_pr(self, metrics: Dict[str, Any], save_path: str):
		fpr, tpr = metrics.get('roc_curve', ([], []))
		prec, rec = metrics.get('pr_curve', ([], []))

		if not fpr or not tpr or not prec or not rec:
			# 单类情况：输出说明性图片
			fig = plt.figure(figsize=(6, 3.2))
			txt = metrics.get('note', 'single-class; ROC/PR unavailable')
			plt.axis('off')
			plt.text(0.02, 0.6, f"ROC/PR 无法计算\n原因: {txt}", fontsize=12)
			fig.tight_layout()
			fig.savefig(save_path, dpi=150)
			plt.close(fig)
			return

		fig, axs = plt.subplots(1, 2, figsize=(10, 4))
		axs[0].plot(fpr, tpr, label=f"ROC AUC={metrics['roc_auc']:.3f}")
		axs[0].plot([0, 1], [0, 1], '--', color='gray')
		axs[0].set_title('ROC')
		axs[0].set_xlabel('FPR')
		axs[0].set_ylabel('TPR')
		axs[0].legend()

		axs[1].plot(rec, prec, label=f"PR AUC={metrics['pr_auc']:.3f}")
		axs[1].set_title('PR')
		axs[1].set_xlabel('Recall')
		axs[1].set_ylabel('Precision')
		axs[1].legend()

		fig.tight_layout()
		fig.savefig(save_path, dpi=150)
		plt.close(fig)

	def summarize(self, metrics: Dict[str, Any], intervention_report: Dict[str, Any]):
		if metrics.get('roc_auc') is None:
			print("单一类别子集：使用无监督异常检测，ROC/PR 不适用。")
		else:
			print(f"ROC AUC: {metrics['roc_auc']:.4f}")
			print(f"PR  AUC: {metrics['pr_auc']:.4f}")
		print("干预报告:")
		for k, v in intervention_report.items():
			print(f"- {k}: {v}")
