from enum import StrEnum
from typing import Callable, Dict, List, TypedDict


class EvaluationMetric(StrEnum):
    TRUSTWORTHINESS = "trustworthiness"
    CONTINUITY = "continuity"
    CA_TRUSTWORTHINESS = "ca_trustworthiness"
    CA_CONTINUITY = "ca_continuity"
    MEAN_RELATIVE_RANK_PRESERVATION_ERROR_FALSE = "mrr_preservation_error_false"
    MEAN_RELATIVE_RANK_PRESERVATION_ERROR_MISSING = "mrr_preservation_error_missing"
    LCMC = "lcmc"
    SPEARMAN_RANK = "spearman_rank"
    SILHOUETTE = "silhouette_score"
    CLUSTER_ARI = "cluster_ari"
    COMPOSITE = "composite"


class MetricConfig(TypedDict):
    label: str
    normalizer: Callable[[float], float]
    weight: float
    enabled: bool


METRIC_CONFIG: Dict[str, MetricConfig] = {
    EvaluationMetric.TRUSTWORTHINESS: {
        "label": "Trustworthiness [0.5,1] (BEST=1)",
        "normalizer": lambda v: (v - 0.5) / 0.5,
        "weight": 1.0,
        "enabled": True,
    },
    EvaluationMetric.CONTINUITY: {
        "label": "Continuity [0.5,1] (BEST=1)",
        "normalizer": lambda v: (v - 0.5) / 0.5,
        "weight": 1.0,
        "enabled": True,
    },
    EvaluationMetric.CA_TRUSTWORTHINESS: {
        "label": "CA Trustworthiness [0.5,1] (BEST=1)",
        "normalizer": lambda v: (v - 0.5) / 0.5,
        "weight": 2.0,
        "enabled": True,
    },
    EvaluationMetric.CA_CONTINUITY: {
        "label": "CA Continuity [0.5,1] (BEST=1)",
        "normalizer": lambda v: (v - 0.5) / 0.5,
        "weight": 2.0,
        "enabled": True,
    },
    EvaluationMetric.LCMC: {
        "label": "LCMC [0,1] (BEST=1)",
        "normalizer": lambda v: v,
        "weight": 1.0,
        "enabled": True,
    },
    EvaluationMetric.SPEARMAN_RANK: {
        "label": "Spearman's rho [-1,1] (BEST=1)",
        "normalizer": lambda v: (v + 1) / 2,
        "weight": 3.0,
        "enabled": True,
    },
    EvaluationMetric.SILHOUETTE: {
        "label": "Silhouette Score [-1,1] (BEST=1)",
        "normalizer": lambda v: (v + 1) / 2,
        "weight": 1.0,
        "enabled": True,
    },
    EvaluationMetric.CLUSTER_ARI: {
        "label": "Cluster ARI [-1,1] (BEST=1)",
        "normalizer": lambda v: (v + 1) / 2,
        "weight": 1.0,
        "enabled": True,
    },
    EvaluationMetric.MEAN_RELATIVE_RANK_PRESERVATION_ERROR_FALSE: {
        "label": "MRRE False [0,1] (BEST=0)",
        "normalizer": lambda v: 1 - v,
        "weight": 1.0,
        "enabled": True,
    },
    EvaluationMetric.MEAN_RELATIVE_RANK_PRESERVATION_ERROR_MISSING: {
        "label": "MRRE Missing [0,1] (BEST=0)",
        "normalizer": lambda v: 1 - v,
        "weight": 1.0,
        "enabled": True,
    },
}


def build_ca_metric_key(metric: EvaluationMetric, column: str) -> str:
    return f"{metric}[{column}]"


def get_metrics_labels(ca_classes: List[str]) -> Dict[str, str]:
    base: Dict[str, str] = {}
    for metric, config in METRIC_CONFIG.items():
        base[metric] = f"{config['label']} W={config['weight']}"

    for column in ca_classes:
        base[build_ca_metric_key(EvaluationMetric.CA_TRUSTWORTHINESS, column)] = (
            f"CA [{column}] Trustworthiness [0.5,1] (BEST=1) W={METRIC_CONFIG[EvaluationMetric.CA_TRUSTWORTHINESS]['weight']}"
        )
        base[build_ca_metric_key(EvaluationMetric.CA_CONTINUITY, column)] = (
            f"CA [{column}] Continuity [0.5,1] (BEST=1) W={METRIC_CONFIG[EvaluationMetric.CA_CONTINUITY]['weight']}"
        )
    return base


def print_metric_results(results: Dict[str, float]) -> None:
    scores = {key: value for key, value in sorted(results.items())}
    print("[Metrics]")
    for key, value in scores.items():
        print(f"  {key:.<60} {value:.6f}")
