"""
cellforge — JEPA stub (Joint Embedding Predictive Architecture).

Real JEPA is a multi-scale predictive model that forecasts future embeddings
from past observations. For v0.3, we ship a SIMPLE LINEAR PREDICTOR as stub:

  - Input: list of past (tick, value) pairs for a cell
  - Output: PredictionCell with {value: probability} distribution
  - The distribution is over plausible next values, NOT a point estimate

The stub uses exponential smoothing with bounded uncertainty. When real JEPA
landships in v0.5+, this stub stays as the deterministic / oracle mode.
"""
import math
import statistics
from typing import Dict, List, Optional, Tuple

from .cells import PredictionCell


class JEPAPredictor:
    """Simple predictor: extrapolates from recent observations with bounded uncertainty.

    Methods:
      - update(tick, value): record an observation
      - predict(horizon): forecast `horizon` ticks ahead
      - confidence(): how sure is this predictor (based on data variance)?
    """

    def __init__(self, source_cell: str, smoothing_alpha: float = 0.5,
                 n_bins: int = 5, value_range: Optional[Tuple[float, float]] = None):
        self.source_cell = source_cell
        self.alpha = smoothing_alpha
        self.n_bins = n_bins
        self.value_range = value_range  # (min, max); inferred from data if not given
        self._observations: List[Tuple[int, float]] = []
        self._smoothed_value: Optional[float] = None

    def update(self, tick: int, value: float) -> None:
        """Record an observation at tick `tick`."""
        self._observations.append((tick, value))
        # Update smoothed value
        if self._smoothed_value is None:
            self._smoothed_value = value
        else:
            self._smoothed_value = (self.alpha * value
                                    + (1 - self.alpha) * self._smoothed_value)
        # Infer range
        if self.value_range is None and len(self._observations) >= 2:
            vals = [v for _, v in self._observations]
            self.value_range = (min(vals) - abs(min(vals)) * 0.1,
                                max(vals) + abs(max(vals)) * 0.1)

    def predict(self, horizon: int = 1, tick: int = 0) -> PredictionCell:
        """Forecast `horizon` ticks ahead; return distribution over plausible values.

        Uses simple linear extrapolation with variance-based uncertainty.
        """
        if self._smoothed_value is None or len(self._observations) < 2:
            return PredictionCell(
                prediction_id=f"pred_{self.source_cell}_t{tick}_h{horizon}",
                source_cell=self.source_cell,
                distribution={str(self._smoothed_value or 0.0): 1.0},
                horizon=horizon,
                confidence=0.0,  # no data, no confidence
                tick=tick,
            )

        # Compute linear slope from recent observations
        recent = self._observations[-min(5, len(self._observations)):]
        if len(recent) >= 2:
            xs = [t for t, _ in recent]
            ys = [v for _, v in recent]
            slope = (ys[-1] - ys[0]) / max(xs[-1] - xs[0], 1)
        else:
            slope = 0.0
        mean_pred = self._smoothed_value + slope * horizon

        # Compute variance for uncertainty bound
        vals = [v for _, v in self._observations]
        if len(vals) >= 2:
            std = statistics.pstdev(vals) * (1 + 0.2 * horizon)  # uncertainty grows with horizon
        else:
            std = 0.1

        # Build distribution over n_bins
        lo = mean_pred - 2 * std
        hi = mean_pred + 2 * std
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        step = (hi - lo) / self.n_bins
        distribution = {}
        for i in range(self.n_bins):
            v = lo + i * step
            # Gaussian-ish weights peaked at mean
            z = (v - mean_pred) / max(std, 1e-9)
            w = math.exp(-0.5 * z * z)
            distribution[str(round(v, 3))] = w
        # Normalize
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}

        # Confidence: lower when std is high relative to mean magnitude
        conf = max(0.0, min(1.0, 1.0 - abs(std) / max(abs(mean_pred), 1.0)))

        return PredictionCell(
            prediction_id=f"pred_{self.source_cell}_t{tick}_h{horizon}",
            source_cell=self.source_cell,
            distribution=distribution,
            horizon=horizon,
            confidence=conf,
            tick=tick,
        )

    def confidence(self) -> float:
        """Static confidence estimate based on variance."""
        if len(self._observations) < 2:
            return 0.0
        vals = [v for _, v in self._observations]
        std = statistics.pstdev(vals)
        mean = statistics.mean(vals)
        return max(0.0, min(1.0, 1.0 - abs(std) / max(abs(mean), 1.0)))


class JEVVerifier:
    """Verifies a PredictionCell against observed canon.

    Simple v0.3 stub: cosine similarity in value-space.
    Real JEV (Joint Embedding Validator) at api.typesafe.ai/v1/systemone
    is the production oracle; this stub is for offline testing.

    Returns a score in [0, 1] measuring agreement with observed value.
    """

    def __init__(self, agreement_threshold: float = 0.7):
        self.agreement_threshold = agreement_threshold

    def verify(self, prediction: PredictionCell, observed_value: float) -> float:
        """Score the prediction against the observed value.

        Returns the probability mass around the observed value.
        """
        if not prediction.distribution:
            return 0.0
        # Find the bin closest to observed_value
        best_mass = 0.0
        for v_str, mass in prediction.distribution.items():
            try:
                v = float(v_str)
            except ValueError:
                continue
            if abs(v - observed_value) < 0.1:  # close enough
                best_mass = max(best_mass, mass)
        return best_mass

    def should_promote(self, prediction: PredictionCell, observed_value: float) -> bool:
        """Should the prediction be promoted to canon?"""
        return self.verify(prediction, observed_value) >= self.agreement_threshold
