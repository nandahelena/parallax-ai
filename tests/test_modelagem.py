import numpy as np
import pandas as pd

from parallax.data.history import channel_history
from parallax.data.splits import channel_folds, temporal_split
from parallax.model.metrics import evaluate, pairwise_acc


def _posts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "channel_id": rng.choice(["a", "b", "c"], n),
        "published_at": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(rng.permutation(n * 7)[:n], "h"),
        "metric_target": rng.integers(0, 10_000, n),
    })


def test_historico_nao_enxerga_o_proprio_post_nem_o_futuro():
    df = _posts()
    corte = df["published_at"].median()
    futuro = df.copy()
    futuro.loc[futuro["published_at"] >= corte, "metric_target"] *= 1000  # mexe no alvo e no futuro
    antes = df["published_at"] <= corte
    pd.testing.assert_frame_equal(channel_history(df)[antes], channel_history(futuro)[antes])


def test_mediana_bate_com_forca_bruta():
    df = _posts()
    h = channel_history(df)
    l = np.log1p(df["metric_target"].astype(float))
    for i in df.sample(30, random_state=1).index:
        mask = (df["channel_id"] == df.at[i, "channel_id"]) & (df["published_at"] < df.at[i, "published_at"])
        anteriores = l[mask].loc[df.loc[mask, "published_at"].sort_values().index].tail(20)
        assert h.at[i, "ch_n_prev"] == mask.sum()
        assert np.isclose(h.at[i, "ch_med_prev20"], anteriores.median(), equal_nan=True)


def test_splits_sem_sobreposicao():
    df = _posts()
    s = temporal_split(df["published_at"])
    t = df.groupby(s)["published_at"]
    assert t.max()["train"] < t.min()["val"] and t.max()["val"] < t.min()["test"]
    f = channel_folds(df["channel_id"], n=3)
    assert df.groupby("channel_id").apply(lambda g: len(set(f[df.index.get_indexer(g.index)]))).eq(1).all()


def test_metricas_casos_conhecidos():
    df = pd.DataFrame({"channel_id": ["a"] * 6, "y": np.arange(6.0)})
    perfeito = evaluate(df.assign(p=df["y"]))
    assert perfeito["mae_log"] == 0 and perfeito["spearman_intra"] == 1 and perfeito["pairwise_intra"] == 1
    assert pairwise_acc(df.assign(p=-df["y"])) == 0
    assert pairwise_acc(df.assign(p=0.0)) == 0.5  # previsão constante = moeda


def test_cqr_corrige_intervalo_estreito():
    from parallax.model.intervals import cqr_offset
    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(size=4000), rng.normal(size=4000)
    lo, hi = -0.5, 0.5  # intervalo estreito demais: cobre só ~38%
    q = cqr_offset(y_cal, np.full(4000, lo), np.full(4000, hi), alpha=0.2)
    cobertura = ((y_new >= lo - q) & (y_new <= hi + q)).mean()
    assert q > 0 and abs(cobertura - 0.8) < 0.03
