from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class SimulationConfig:
	num_users: int = 2000
	fraud_ratio: float = 0.08
	seed: int = 42


@dataclass
class FeatureConfig:
	ngram_range: Tuple[int, int] = (1, 2)
	max_features: int = 5000


@dataclass
class ModelConfig:
	type: str = "xgb_like"
	params: Dict[str, Any] = None


@dataclass
class InterventionConfig:
	threshold: float = 0.7
	cooldown_days: int = 7
	max_daily_flags: int = 100


@dataclass
class EvaluationConfig:
	test_size: float = 0.25
	random_state: int = 42
