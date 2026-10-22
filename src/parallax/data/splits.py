"""Splits honestos (PLANO §5): temporal como principal, por canal para "cliente novo". Nunca CV aleatória."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def temporal_split(published_at: pd.Series, fracs: tuple[float, float] = (0.70, 0.85)) -> pd.Series:
    """train / val / test por data de publicação: o val (fatia mais recente antes do teste) serve para
    early stopping e calibração conformal."""
    c1, c2 = published_at.quantile(list(fracs))
    return pd.Series(np.select([published_at < c1, published_at < c2], ["train", "val"], "test"),
                     index=published_at.index)


def channel_folds(channel_id: pd.Series, n: int = 5) -> np.ndarray:
    """Índice de fold por linha; cada canal cai inteiro em um fold."""
    fold = np.empty(len(channel_id), dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n).split(channel_id, groups=channel_id)):
        fold[te] = k
    return fold
