"""Features do histórico do canal usando só posts anteriores ao alvo (PLANO regra 4).

Também serve para a inferência: concatenar o histórico do canal com o post novo (metric_target ausente)
e ler a última linha, pelo mesmo código do treino.
"""

import numpy as np
import pandas as pd

from parallax import config


def channel_history(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe channel_id, published_at (UTC) e metric_target; devolve features alinhadas ao índice de `df`.

    shortcut: dois posts do mesmo canal com o mesmo timestamp exato se enxergam via shift; é raríssimo em Shorts.
    """
    d = df[["channel_id", "published_at", "metric_target"]].sort_values(["channel_id", "published_at"])
    canal = d["channel_id"]
    prev = np.log1p(d["metric_target"].astype(float)).groupby(canal).shift(1)  # o próprio post nunca entra
    janela = lambda n, f, mp=1: prev.groupby(canal).transform(lambda s: getattr(s.rolling(n, min_periods=mp), f)())  # noqa: E731

    out = pd.DataFrame(index=d.index)
    out["ch_n_prev"] = d.groupby("channel_id").cumcount()
    out["ch_med_prev20"] = janela(config.K_HISTORICO, "median")
    out["ch_mean_prev5"] = janela(5, "mean")
    out["ch_mean_prev20"] = janela(config.K_HISTORICO, "mean")
    out["ch_std_prev20"] = janela(config.K_HISTORICO, "std", 2)
    out["ch_trend"] = out["ch_mean_prev5"] - out["ch_mean_prev20"]
    out["ch_last"] = prev
    out["ch_days_since_last"] = d.groupby("channel_id")["published_at"].diff().dt.total_seconds() / 86400
    out["ch_posts_30d"] = np.concatenate([
        pd.Series(1.0, index=g["published_at"].to_numpy()).rolling("30D", closed="left").sum().fillna(0).to_numpy()
        for _, g in d.groupby("channel_id", sort=True)
    ])
    return out.reindex(df.index)
