"""Filtros da base (PLANO.md F1.5, decisão D02)."""

import numpy as np
import pandas as pd

from parallax import config


def shorts_mask(df: pd.DataFrame) -> pd.Series:
    """Heurística de Short (a API não tem `isShort`): vertical e duração dentro do limite vigente na data.

    Até 15/10/2024 o limite era 60 s; depois, 180 s. Sem dimensões do player, a orientação é desconhecida
    e o vídeo fica de fora (conservador).
    """
    pub = pd.to_datetime(df["published_at"], utc=True)
    limite = np.where(pub < pd.Timestamp(config.SHORTS_REGRA_MUDOU_EM, tz="UTC"),
                      config.SHORTS_MAX_S_ANTES, config.SHORTS_MAX_S_DEPOIS)
    vertical = df["embed_h"] > df["embed_w"]
    return (df["duration_s"] <= limite) & vertical.fillna(False).astype(bool)
