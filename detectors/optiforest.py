from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest

class OptIForest:
    """Minimal OptIForest-compatible API using IsolationForest under the hood.
    This mirrors the xiagll/OptIForest interface so you can swap in the real algorithm later.
    """
    def __init__(self, n_estimators: int = 200, branch: int | None = None, random_state: int | None = 42, contamination: float | None = None, n_jobs: int | None = -1):
        self.n_estimators = n_estimators
        self.branch = branch  # kept for CLI parity; unused by sklearn backend
        self.random_state = random_state
        self.contamination = contamination
        self.n_jobs = n_jobs
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._fitted = False

    def fit(self, X: np.ndarray):
        self._model.fit(X)
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("OptIForest is not fitted. Call fit(X) first.")
        # sklearn IF: lower score_samples => more anomalous. Flip sign so HIGHER = MORE anomalous.
        return -self._model.score_samples(X)
