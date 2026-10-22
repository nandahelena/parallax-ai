"""Peças de treino compartilhadas por 07_treinar (Opção A) e 08_ablacao (multimodal)."""

import lightgbm as lgb  # antes de qualquer torch (conflito de OpenMP, troubleshooting da liga)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from parallax import config
from parallax.data.splits import temporal_split
from parallax.features.build import tabular_features
from parallax.features.text import text_embeddings

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_estimators=5000,
              random_state=config.SEED, verbose=-1)
PCA_COLS = [f"tx_pca_{i:02d}" for i in range(config.N_PCA_TEXTO)]
BASE_VERSAO = "base_v2 (coleta 15/10, estatísticas recoletadas em 16/10)"


def add_pca(X: pd.DataFrame, E: np.ndarray, fit: np.ndarray, prefixo: str = "tx_pca",
            n: int = config.N_PCA_TEXTO) -> tuple[pd.DataFrame, PCA]:
    """Componentes de PCA de um embedding, ajustado só nas linhas `fit` (PLANO regra 8)."""
    nomes = [f"{prefixo}_{i:02d}" for i in range(n)]
    pca = PCA(n, random_state=config.SEED).fit(E[fit])
    return X.drop(columns=nomes, errors="ignore").join(pd.DataFrame(pca.transform(E), columns=nomes, index=X.index)), pca


def fit_lgb(X, target, tr, va=None, n_estimators=None, init_score=None, **extra):
    m = lgb.LGBMRegressor(**{**PARAMS, **({"n_estimators": n_estimators} if n_estimators else {}), **extra})
    kw = {}
    if init_score is not None:
        kw["init_score"] = init_score[tr]
    if va is not None:
        kw.update(eval_set=[(X[va], target[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
        if init_score is not None:
            kw["eval_init_score"] = [init_score[va]]
    return m.fit(X[tr], target[tr], **kw)


def carregar_treino():
    """Canais de treino com histórico mínimo: base, features tabulares, alvos, embedding de texto e split temporal."""
    b = pd.read_parquet(config.PROCESSED / config.BASE_TREINO)
    b = b[b["role"] == "treino"].reset_index(drop=True)  # demo fica fora de tudo
    X = tabular_features(b)
    keep = X["ch_n_prev"] >= config.MIN_HISTORICO  # sem histórico mínimo não há alvo relativo
    b, X = b[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    E = text_embeddings(b["post_id"], b["title"], b["description"])
    y = np.log1p(b["metric_target"].astype(float))
    return b, X, y, E, temporal_split(b["published_at"])
