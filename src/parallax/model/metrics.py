"""Métricas do PLANO §5, todas em escala log, com IC por bootstrap de canais."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def pairwise_acc(df: pd.DataFrame, margin: float = 0.0) -> float:
    """Acurácia par a par dentro de cada canal: o modelo ordena certo os dois criativos?
    Pares com |Δy| <= margin são ignorados; empate na previsão vale meio acerto."""
    acertos = total = 0.0
    for _, g in df.groupby("channel_id"):
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        iu = np.triu_indices(len(y), 1)
        dy, dp = (y[:, None] - y[None, :])[iu], (p[:, None] - p[None, :])[iu]
        ok = np.abs(dy) > margin
        acertos += ((dy * dp > 0) + 0.5 * (dp == 0))[ok].sum()
        total += ok.sum()
    return acertos / total if total else np.nan


def spearman_intra(df: pd.DataFrame, min_n: int = 5) -> float:
    rs = [spearmanr(g["y"], g["p"]).statistic for _, g in df.groupby("channel_id") if len(g) >= min_n]
    return float(np.nanmean(rs))


def evaluate(df: pd.DataFrame) -> dict:
    """df: y (log1p real), p (log1p previsto), channel_id."""
    err = (df["p"] - df["y"]).abs()
    pos = df["y"] > 0
    return {
        "mae_log": err.mean(),
        "mape_log": (err[pos] / df.loc[pos, "y"]).mean(),
        "spearman_global": spearmanr(df["y"], df["p"]).statistic,
        "spearman_intra": spearman_intra(df),
        "pairwise_intra": pairwise_acc(df),
        "pairwise_intra_m01": pairwise_acc(df, 0.1),
    }


def bootstrap(dfs: dict[str, pd.DataFrame], n: int = 300, seed: int = 42) -> dict:
    """IC95% reamostrando canais (as mesmas reamostras para todos os modelos, logo Δs são pareados).
    dfs: {modelo: df com y, p, channel_id}, todos sobre as mesmas linhas."""
    rng = np.random.default_rng(seed)
    canais = next(iter(dfs.values()))["channel_id"].unique()
    idx = {m: d.groupby("channel_id").indices for m, d in dfs.items()}
    amostras = {m: [] for m in dfs}
    for _ in range(n):
        escolha = rng.choice(canais, len(canais))
        for m, d in dfs.items():
            linhas = np.concatenate([idx[m][c] for c in escolha])
            # canais repetidos viram canais distintos, senão os pares intra-canal se misturam
            rep = d.iloc[linhas].assign(channel_id=np.repeat(np.arange(len(escolha)),
                                                             [len(idx[m][c]) for c in escolha]))
            amostras[m].append(evaluate(rep))
    return {m: pd.DataFrame(a) for m, a in amostras.items()}


def ci(s: pd.Series) -> str:
    return f"[{s.quantile(0.025):.3f}, {s.quantile(0.975):.3f}]"
