"""
Post-Training Temperature Calibration for ApexStudent.
Optimizes scaling temperatures per (question_type, option_bucket) to minimize
Expected Calibration Error (ECE) and Brier score on held-out calibration data.
"""

import math
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


def temp_bucket(qtype: int, k: int) -> str:
    """Bucket by question type and option cardinality."""
    qnames = {0: "choice", 1: "score", 2: "noul", 3: "verify"}
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11-20" if k <= 20 else "21-50" if k <= 50 else "51+"
    return f"{qnames.get(int(qtype), 'choice')}:{size}"


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Computes Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def fit_temperature(logits_list: List[np.ndarray], targets_list: List[np.ndarray]) -> float:
    """
    Fits single temperature scalar T > 0 by minimizing NLL.
    """
    if not logits_list:
        return 1.0

    def nll_obj(log_t):
        T = math.exp(log_t[0])
        total_nll = 0.0
        for z, y in zip(logits_list, targets_list):
            k = len(z)
            scaled_z = z / T
            p = np.exp(scaled_z - np.max(scaled_z))
            p = p / np.sum(p)
            log_p = np.log(np.clip(p, 1e-12, 1.0))
            total_nll -= float(np.sum(y[:k] * log_p))
        return total_nll / len(logits_list)

    res = minimize(nll_obj, [0.0], method="Nelder-Mead")
    fitted_t = math.exp(res.x[0])
    return float(np.clip(fitted_t, 0.5, 2.0))


class TemperatureCalibrator:
    """
    Fits and stores calibrated temperatures per bucket.
    """
    def __init__(self):
        self.bucket_temperatures: Dict[str, float] = {}
        self.default_temperature = 1.0

    def fit(self, calibration_records: List[Dict[str, Any]]):
        """
        calibration_records: list of dicts with 'qtype', 'k', 'logits', 'target'
        """
        buckets: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
        for rec in calibration_records:
            b_key = temp_bucket(rec["qtype"], rec["k"])
            if b_key not in buckets:
                buckets[b_key] = ([], [])
            buckets[b_key][0].append(np.asarray(rec["logits"], dtype=np.float32))
            buckets[b_key][1].append(np.asarray(rec["target"], dtype=np.float32))

        for b_key, (logits, targets) in buckets.items():
            self.bucket_temperatures[b_key] = fit_temperature(logits, targets)

    def get_temperature(self, qtype: int, k: int) -> float:
        b_key = temp_bucket(qtype, k)
        return self.bucket_temperatures.get(b_key, self.default_temperature)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_temperature": self.default_temperature,
            "bucket_temperatures": self.bucket_temperatures
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemperatureCalibrator":
        inst = cls()
        inst.default_temperature = data.get("default_temperature", 1.0)
        inst.bucket_temperatures = data.get("bucket_temperatures", {})
        return inst
