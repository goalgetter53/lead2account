"""Logistic-regression pair scorer, stored as plain JSON (no pickle)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from l2a.features import FEATURES


@dataclass
class PairModel:
    coef: NDArray[np.float64]
    intercept: float
    threshold: float = 0.5
    features: tuple[str, ...] = FEATURES

    @classmethod
    def fit(cls, x: NDArray[np.float64], y: NDArray[np.int_], c: float = 1.0) -> PairModel:
        lr = LogisticRegression(C=c, max_iter=2000)
        lr.fit(x, y)
        return cls(coef=lr.coef_[0].astype(np.float64), intercept=float(lr.intercept_[0]))

    def proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = x @ self.coef + self.intercept
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)

    def tune_threshold(self, p: NDArray[np.float64], y: NDArray[np.int_]) -> float:
        """Pick the threshold maximising pairwise F1 (dev data only)."""
        best_t, best_f = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            pred = p >= t
            tp = float(np.sum(pred & (y == 1)))
            fp = float(np.sum(pred & (y == 0)))
            fn = float(np.sum(~pred & (y == 1)))
            f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f > best_f:
                best_t, best_f = float(t), f
        self.threshold = best_t
        return best_t

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "features": list(self.features),
                    "coef": [round(float(v), 6) for v in self.coef],
                    "intercept": round(self.intercept, 6),
                    "threshold": self.threshold,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, path: Path) -> PairModel:
        d = json.loads(path.read_text())
        if tuple(d["features"]) != FEATURES:
            raise ValueError("feature list changed; retrain the model")
        return cls(
            coef=np.asarray(d["coef"], dtype=np.float64),
            intercept=float(d["intercept"]),
            threshold=float(d["threshold"]),
        )
