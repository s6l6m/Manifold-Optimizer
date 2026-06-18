from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Mapping, Sequence

import optuna
from sklearn.model_selection import ParameterGrid

import numpy as np


@dataclass
class Projector(ABC):
    """Abstract base class for dimensionality reduction methods using a specific implementation."""

    parameter_map: ClassVar[Mapping[str, Sequence[Any]]] = {}

    def get_parameter_grid(self) -> ParameterGrid:
        return ParameterGrid(self.parameter_map)

    @abstractmethod
    def suggest_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Declare the Bayesian optimisation search space via trial.suggest_* calls."""

    @abstractmethod
    def project(
        self,
        data: np.ndarray,
        subset_index_array: np.ndarray,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute or retrieve the reduced embedding

        Parameters
        ----------
        data   : full dataset
        subset_index_array : integer index array selecting the subset
        params : parameter combination to compute the embeddings

        Returns the reduced embedding for `data[subset_index_array]` only.
        """
