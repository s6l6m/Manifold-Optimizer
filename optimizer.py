from config import Config
from metrics import EvaluationMetric, get_metrics_labels
from scorer import Scorer
from projector import Projector
import numpy as np
import pandas as pd
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import math
from typing import Any, Dict, List, Optional


class Optimizer:
    def __init__(
        self,
        projector: "Projector",
        data: np.ndarray,
        subset_indices: np.ndarray,
        compound_metadata: pd.DataFrame,
        experiment_name: str = "Experiment",
        ca_classes: List[str] = [],
        config: Optional[Config] = None,
    ):
        self.projector = projector
        self.data = data
        self.subset = subset_indices
        self.experiment_name = experiment_name
        self.ca_classes: List[str] = ca_classes
        self.results_df: Optional[pd.DataFrame] = None
        self.study: Optional[optuna.Study] = None
        self.config = config or Config()
        self.compound_metadata = compound_metadata

    def _build_ca_labels(self) -> Dict[str, np.ndarray]:
        return {
            ca_class: self.compound_metadata.loc[self.subset, ca_class]
            .fillna("Unknown")
            .astype(str)
            .to_numpy()
            for ca_class in self.ca_classes
        }

    def _evaluate(
        self,
        params: dict,
        verbose: bool,
    ) -> dict[str, Any]:
        if verbose:
            print(f"Evaluating params: {params}")

        X_hd_subset = self.data[self.subset]
        X_ld_subset = self.projector.project(self.data, self.subset, params)

        assert X_ld_subset.shape[0] == X_hd_subset.shape[0], (
            "Subset size mismatch after projection. Make sure the projector returns the correct number of samples for the given subset."
        )

        scorer = Scorer(self.config)
        scores = scorer.evaluate_embedding_quality(
            X_hd=X_hd_subset,
            X_ld=X_ld_subset,
            ca_labels=self._build_ca_labels(),
            verbose=verbose,
        )
        return {
            "scores": scores,
            "embedding": X_ld_subset,
            "composite": scorer.composite_score(scores),
        }

    def _run_study(
        self,
        sampler: optuna.samplers.BaseSampler,
        n_trials: int,
        verbose: bool,
    ) -> pd.DataFrame:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = self.projector.suggest_params(trial)

            for t in trial.study.trials:
                if (
                    t.number != trial.number
                    and t.state == optuna.trial.TrialState.COMPLETE
                    and t.params == trial.params
                    and t.value is not None
                ):
                    for key, value in t.user_attrs.items():
                        trial.set_user_attr(key, value)
                    return t.value

            results = self._evaluate(params, verbose)
            for key, value in results["scores"].items():
                trial.set_user_attr(str(key), value)
            return results["composite"]

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=self.experiment_name,
            storage=self.config.OPTUNA_SQLITE,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.study = study
        self.results_df = self._build_results_df(study)
        return self.results_df

    def _build_results_df(self, study: optuna.Study) -> pd.DataFrame:
        scorer = Scorer(self.config)
        rows = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE or not trial.user_attrs:
                continue
            rows.append({
                **trial.params,
                **trial.user_attrs,
                "composite": scorer.composite_score(trial.user_attrs),
            })
        return pd.DataFrame(rows).drop_duplicates()

    def load_study_if_exists(self):
        try:
            study = optuna.load_study(
                study_name=self.experiment_name,
                storage=self.config.OPTUNA_SQLITE,
            )
            self.study = study
            self.results_df = self._build_results_df(study)
            return self.results_df

        except KeyError:
            print(
                f"No existing study found for '{self.experiment_name}'. Starting a new study."
            )
            self.study = None
            self.results_df = None
            return None

    def run_grid_search(self, verbose: bool = False) -> pd.DataFrame:
        param_map = {
            key: list(value) for key, value in self.projector.parameter_map.items()
        }
        n_trials = math.prod(len(value) for value in param_map.values())
        return self._run_study(
            sampler=optuna.samplers.GridSampler(param_map),
            n_trials=n_trials,
            verbose=verbose,
        )

    def run_optimizer(
        self,
        n_iterations: int = 500,
        verbose: bool = False,
    ) -> pd.DataFrame:
        return self._run_study(
            sampler=optuna.samplers.GPSampler(seed=self.config.RANDOM_SEED),
            n_trials=n_iterations,
            verbose=verbose,
        )

    def get_best_params(self, metric: str = EvaluationMetric.COMPOSITE, high_is_better: bool = False):
        assert self.study is not None and self.results_df is not None, "Study results should be available after running the experiment."
        metric_map = get_metrics_labels(self.ca_classes)
        param_cols = [
            column for column in self.results_df.columns if column not in metric_map and column != "composite"
        ]
        best_row = self.results_df.loc[self.results_df[metric].idxmax() if high_is_better else self.results_df[metric].idxmin()]
        best_row_params = best_row[param_cols].to_dict()
        best_row_params["top_by"] = metric
        return best_row_params

    def plot_statistics(
        self, line_parameter: str, best_parameter, excluded_parameters: Optional[List[str]] = None
    ):
        assert self.results_df is not None, "Please run the experiment first."
        assert self.study is not None, (
            "Study should be available after running the experiment."
        )

        sns.set_theme(style="whitegrid")

        metric_map = get_metrics_labels(self.ca_classes)
        metric_cols = [
            key for key in metric_map.keys() if key in self.results_df.columns
        ]
        param_cols = [
            column for column in self.results_df.columns if column not in metric_map and column != "composite"
        ]

        if not metric_cols or not param_cols:
            return None

        if line_parameter in param_cols:
            x_params = [p for p in param_cols if p != line_parameter]
            line_values = sorted(
                self.results_df[line_parameter].dropna().unique().tolist()
            )
        else:
            x_params = param_cols
            line_values = [None]

        x_params = [p for p in x_params if p not in (excluded_parameters or [])]

        if not x_params:
            return None

        n_rows, n_cols = len(metric_cols), len(x_params)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"{self.experiment_name} [best:{best_parameter}]",
            fontsize=8,
            y=1,
        )

        palette = sns.color_palette("tab10", n_colors=max(len(line_values), 1))

        for row_idx, metric in enumerate(metric_cols):
            for col_idx, param in enumerate(x_params):
                ax = axes[row_idx, col_idx]

                for line_value, color in zip(line_values, palette):
                    sub = (
                        self.results_df
                        if line_value is None
                        else self.results_df[
                            self.results_df[line_parameter] == line_value
                        ]
                    )
                    if sub.empty:
                        continue

                    grouped = (
                        sub.groupby(param, sort=True)[metric]
                        .agg(["mean", "std"])
                        .reset_index()
                    )

                    ax.errorbar(
                        grouped[param],
                        grouped["mean"],
                        yerr=grouped["std"].fillna(0),
                        marker="o",
                        linestyle="--",
                        capsize=3,
                        color=color,
                        label=str(line_value)
                        if line_value is not None
                        else metric_map[metric],
                    )

                ax.set_xlabel(param, fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(metric_map[metric], fontsize=9)

                if len(line_values) > 1:
                    ax.legend(title=line_parameter, fontsize=6, frameon=False)

        plt.tight_layout()
        plt.show()
