"""Calibração conformal dos quantis (CQR, Romano et al. 2019): garante a cobertura nominal em dados novos
sem supor nenhuma distribuição. Feito à mão (5 linhas) para controlar o split temporal: a calibração usa a fatia
de validação, que vem antes do teste no tempo (o helper do MAPIE sorteia aleatoriamente)."""

import numpy as np


def cqr_offset(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.2) -> float:
    """Quanto alargar (q > 0) ou estreitar (q < 0) [lo, hi] para cobrir 1 - alpha na calibração."""
    s = np.maximum(np.asarray(lo) - y, np.asarray(y) - hi)
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
