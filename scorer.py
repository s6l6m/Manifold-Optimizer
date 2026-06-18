from typing import Dict, Optional, Tuple

from config import Config
from metrics import (
    METRIC_CONFIG,
    EvaluationMetric,
    print_metric_results,
    build_ca_metric_key,
)

import cupy as cp
from cuml.cluster import KMeans as cuKMeans
from cuml.metrics import pairwise_distances
from cuml.metrics.cluster import silhouette_score, adjusted_rand_score
from zadu.measures import (
    trustworthiness_continuity,
    mean_relative_rank_error,
    class_aware_trustworthiness_continuity,
    local_continuity_meta_criteria,
    spearman_rho,
)

import numpy as np

class Scorer:
    """Scoring class to evaluate the quality of dimensionality reduction embeddings using various metrics."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def knn_with_ranking(
        self, d_gpu: "cp.ndarray", k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        sorted_idx = cp.argsort(d_gpu, axis=1)  # GPU row-wise sort
        knn_indices = cp.asnumpy(sorted_idx[:, 1 : k + 1])  # skip self (rank 0)
        ranking = cp.asnumpy(cp.argsort(sorted_idx, axis=1))  # inverse permutation
        return knn_indices, ranking

    def evaluate_embedding_quality(
        self,
        X_hd: np.ndarray,
        X_ld: np.ndarray,
        ca_labels: Optional[Dict[str, np.ndarray]] = None,
        n_clusters: int = 20,
        k_neighbors: int = 20,
        verbose: bool = False,
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}

        X_hd_gpu = cp.asarray(
            X_hd, dtype=cp.float32 if not self.config.BINARY_EMBEDDING else cp.uint8
        )
        X_ld_gpu = cp.asarray(
            X_ld, dtype=cp.float32 if not self.config.BINARY_EMBEDDING else cp.uint8
        )

        labels_hd = cuKMeans(
            n_clusters=n_clusters, random_state=self.config.RANDOM_SEED
        ).fit_predict(X_hd_gpu)
        labels_ld = cuKMeans(
            n_clusters=n_clusters, random_state=self.config.RANDOM_SEED
        ).fit_predict(X_ld_gpu)

        if METRIC_CONFIG[EvaluationMetric.SILHOUETTE]["enabled"]:
            results[EvaluationMetric.SILHOUETTE] = float(
                silhouette_score(X_ld_gpu, labels_ld)
            )

        if METRIC_CONFIG[EvaluationMetric.CLUSTER_ARI]["enabled"]:
            results[EvaluationMetric.CLUSTER_ARI] = float(
                adjusted_rand_score(labels_hd, labels_ld)
            )

        d_hd_cp = pairwise_distances(X_hd_gpu)
        d_ld_cp = pairwise_distances(X_ld_gpu)

        hd_knn, hd_ranking = self.knn_with_ranking(d_hd_cp, k_neighbors)
        ld_knn, ld_ranking = self.knn_with_ranking(d_ld_cp, k_neighbors)

        d_hd_np = cp.asnumpy(d_hd_cp)
        d_ld_np = cp.asnumpy(d_ld_cp)
        del d_hd_cp, d_ld_cp  # free GPU memory

        knn_rank_info = (hd_knn, hd_ranking, ld_knn, ld_ranking)

        if METRIC_CONFIG[EvaluationMetric.LCMC]["enabled"]:
            lcmc = local_continuity_meta_criteria.measure(
                orig=X_hd, emb=X_ld, k=k_neighbors, knn_info=(hd_knn, ld_knn)
            )
            if type(lcmc) is dict:
                results[EvaluationMetric.LCMC] = lcmc["lcmc"]

        if METRIC_CONFIG[EvaluationMetric.SPEARMAN_RANK]["enabled"]:
            spr = spearman_rho.measure(
                orig=X_hd, emb=X_ld, distance_matrices=(d_hd_np, d_ld_np)
            )
            if type(spr) is dict:
                results[EvaluationMetric.SPEARMAN_RANK] = spr["spearman_rho"]

        if (
            METRIC_CONFIG[EvaluationMetric.MEAN_RELATIVE_RANK_PRESERVATION_ERROR_FALSE][
                "enabled"
            ]
            or METRIC_CONFIG[
                EvaluationMetric.MEAN_RELATIVE_RANK_PRESERVATION_ERROR_MISSING
            ]["enabled"]
        ):
            mrre = mean_relative_rank_error.measure(
                orig=X_hd_gpu,
                emb=X_ld_gpu,
                k=k_neighbors,
                knn_ranking_info=knn_rank_info,
            )
            if type(mrre) is dict:
                if METRIC_CONFIG[
                    EvaluationMetric.MEAN_RELATIVE_RANK_PRESERVATION_ERROR_FALSE
                ]["enabled"]:
                    results[
                        EvaluationMetric.MEAN_RELATIVE_RANK_PRESERVATION_ERROR_FALSE
                    ] = mrre["mrre_false"]

                if METRIC_CONFIG[
                    EvaluationMetric.MEAN_RELATIVE_RANK_PRESERVATION_ERROR_MISSING
                ]["enabled"]:
                    results[
                        EvaluationMetric.MEAN_RELATIVE_RANK_PRESERVATION_ERROR_MISSING
                    ] = mrre["mrre_missing"]

        if (
            METRIC_CONFIG[EvaluationMetric.TRUSTWORTHINESS]["enabled"]
            or METRIC_CONFIG[EvaluationMetric.CONTINUITY]["enabled"]
        ):
            tnc = trustworthiness_continuity.measure(
                orig=X_hd, emb=X_ld, k=k_neighbors, knn_ranking_info=knn_rank_info
            )
            if type(tnc) is dict:
                if METRIC_CONFIG[EvaluationMetric.TRUSTWORTHINESS]["enabled"]:
                    results[EvaluationMetric.TRUSTWORTHINESS] = tnc["trustworthiness"]

                if METRIC_CONFIG[EvaluationMetric.CONTINUITY]["enabled"]:
                    results[EvaluationMetric.CONTINUITY] = tnc["continuity"]

        if ca_labels:
            for ca_class, labels in ca_labels.items():
                ca_tnc = class_aware_trustworthiness_continuity.measure(
                    orig=X_hd_gpu,
                    emb=X_ld_gpu,
                    label=labels,
                    k=k_neighbors,
                    knn_ranking_info=knn_rank_info,
                )
                if type(ca_tnc) is dict:
                    results[
                        build_ca_metric_key(
                            EvaluationMetric.CA_TRUSTWORTHINESS, ca_class
                        )
                    ] = ca_tnc["ca_trustworthiness"]
                    results[
                        build_ca_metric_key(EvaluationMetric.CA_CONTINUITY, ca_class)
                    ] = ca_tnc["ca_continuity"]

        if verbose:
            print_metric_results(results)

        return results

    def _resolve_base_metric(self, metric: str) -> EvaluationMetric:
        if metric.startswith(EvaluationMetric.CA_TRUSTWORTHINESS):
            return EvaluationMetric.CA_TRUSTWORTHINESS
        if metric.startswith(EvaluationMetric.CA_CONTINUITY):
            return EvaluationMetric.CA_CONTINUITY
        return EvaluationMetric(metric)

    def composite_score(self, scores: Dict[str, float]) -> float:
        if not scores:
            return 0.0

        normalized, weights = [], []
        for metric, value in scores.items():
            base_metric = self._resolve_base_metric(metric)
            if base_metric not in METRIC_CONFIG:
                raise ValueError(f"No config defined for Metric '{metric}'")
            config = METRIC_CONFIG[base_metric]
            normalized.append(config["normalizer"](value))
            weights.append(config["weight"])

        total_weight = sum(weights)
        return (
            float(np.dot(normalized, weights) / total_weight) if total_weight > 0 else 0.0
        )
