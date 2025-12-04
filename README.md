# 基于社会计算的金融诈骗检测机制（代码实现）

## 快速开始
1. 创建虚拟环境并安装依赖：
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.**txt**
```
2. 运行示例流程（数据模拟 → 特征工程 → 社会信号集成 → 训练检测模型 → 干预仿真 → 评估可视化）：
```bash
.venv\Scripts\python main.py --config configs/base.yaml
```

## 使用外部数据集（UCI）
- UCI Phishing Websites / SMS Spam 解压到某目录（例如 `D:\py\third1\Social_Computing\uci`），再运行：
```bash
# UCI Phishing Websites
.venv\Scripts\python main.py --dataset uci_phishing --uci_dir D:\py\third1\Social_Computing

# UCI SMS Spam Collection
.venv\Scripts\python main.py --dataset sms_spam --uci_dir D:\py\third1\Social_Computing
```
- 加载器会在 `--uci_dir` 中查找常见文件名（如 `phishing.csv`、`PhishingData.csv`、`SMSSpamCollection` 等）。找不到会报错提示文件名。

## 功能模块
- 数据模拟与加载：`src/data/simulate.py`, `src/data/dataset.py`, `src/data/loader_uci.py`
- 特征工程（内容/用户/网络/时间）：`src/features/extract.py`
- 社会信号与图计算：`src/signals/social_graph.py`
- 检测模型：`src/models/detector.py`
- 干预策略与仿真：`src/intervention/strategy.py`
- 评估与可视化：`src/evaluation/evaluate.py`

## 数据说明
本项目自带合成数据生成器，同时支持接入 UCI 数据集。由于 UCI 数据不含社交关系，加载器会生成最小用户表与空交互表，仅用于文本/画像特征实验。

## 许可证
MIT
