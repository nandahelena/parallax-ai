"""F2.5–F2.10 (G1): baselines + LightGBM Opção A, avaliados no split temporal (principal), no split por canal
("cliente novo") e, só para medir o otimismo, em CV aleatória. Tudo registrado no MLflow.

    uv run python scripts/07_treinar.py
"""

import json

import joblib

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from parallax import config
from parallax.data.splits import channel_folds
from parallax.features.build import cols
from parallax.model.intervals import cqr_offset
from parallax.model.metrics import bootstrap, ci, evaluate
from parallax.model.train import BASE_VERSAO, PARAMS, PCA_COLS, add_pca, carregar_treino, fit_lgb

OUT = config.ROOT / "docs" / "resultados"
VERSAO_TREINO = "v1"  # versão gerada por este script (o v0 fica preservado)


def oof(Xa, E, y_rel, base_ch, folds, feats, n_est):
    """Previsões fora da dobra (escala log absoluta) com PCA reajustado em cada dobra."""
    p = pd.Series(np.nan, index=Xa.index)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        Xk, _ = add_pca(Xa, E, tr)
        m = fit_lgb(Xk[feats], y_rel, tr, n_estimators=n_est)
        p[te] = base_ch[te] + m.predict(Xk.loc[te, feats])
    return p


def main() -> None:
    b, X, y, E, split = carregar_treino()
    base_ch = X["ch_med_prev20"]
    y_rel = y - base_ch

    tr, va, te = (split == "train").to_numpy(), (split == "val").to_numpy(), (split == "test").to_numpy()
    cortes = {s: [str(b.loc[split == s, "published_at"].min().date()), str(b.loc[split == s, "published_at"].max().date())]
              for s in ("train", "val", "test")}
    Xp, pca = add_pca(X, E, tr)
    feats = {"M1": cols("canal", "tempo", "metadados"),
             "M2": cols("canal", "tempo", "metadados", "texto") + PCA_COLS}
    feats["M2 sem inscritos"] = [c for c in feats["M2"] if c != "ch_log_subs"]

    preds, modelos = {}, {}
    subs_med = X.loc[tr, "ch_log_subs"].median()
    preds["B0 mediana do canal"] = base_ch[te]
    preds["B1 média do canal"] = X.loc[te, "ch_mean_prev20"]
    lin = LinearRegression().fit(X.loc[tr, ["ch_log_subs"]].fillna(subs_med), y[tr])
    preds["B2 linear inscritos"] = pd.Series(lin.predict(X.loc[te, ["ch_log_subs"]].fillna(subs_med)), index=b.index[te])
    preds["B3 mediana do setor"] = b.loc[te, "sector"].map(y[tr].groupby(b.loc[tr, "sector"]).median())
    for nome, fc in feats.items():
        m = modelos[nome] = fit_lgb(Xp[fc], y_rel, tr, va)
        preds[f"{nome} LightGBM"] = base_ch[te] + m.predict(Xp.loc[te, fc])
    # Intervalo de 80% por quantis do GBM (bruto; a calibração conformal entra na F4)
    q = {a: fit_lgb(Xp[feats["M2"]], y_rel, tr, va, objective="quantile", alpha=a) for a in config.QUANTIS}
    def intervalo(mask, aj=0.0):
        """Mesma regra de `inference.prever`: ajusta os quantis e só então reordena em torno do ponto."""
        pr = modelos["M2"].predict(Xp.loc[mask, feats["M2"]])
        lo, hi = (q[a].predict(Xp.loc[mask, feats["M2"]]) for a in config.QUANTIS)
        return np.minimum(lo - aj, pr), np.maximum(hi + aj, pr)
    lo_te, hi_te = intervalo(te)
    cobertura = float(((y_rel[te] >= lo_te) & (y_rel[te] <= hi_te)).mean())
    cob_setor = ((y_rel[te] >= lo_te) & (y_rel[te] <= hi_te)).groupby(b.loc[te, "sector"]).mean().round(3).to_dict()
    lo_va, hi_va = intervalo(va)
    # CQR: calibra no val (fatia temporal antes do teste) e mede no teste. O val também fez early stopping,
    # o que deixa a calibração levemente otimista; a cobertura no teste é a medida honesta.
    q_conf = cqr_offset(y_rel[va].to_numpy(), lo_va, hi_va, alpha=0.2)
    lo_c, hi_c = intervalo(te, q_conf)
    dentro = (y_rel[te] >= lo_c) & (y_rel[te] <= hi_c)
    cobertura_conf = float(dentro.mean())
    cob_setor_conf = dentro.groupby(b.loc[te, "sector"]).mean().round(3).to_dict()
    lo_vc, hi_vc = intervalo(va, q_conf)
    largura_p80 = float(np.quantile(hi_vc - lo_vc, 0.8))
    m_abs = fit_lgb(Xp[feats["M2"]], y, tr, va)
    preds["M2 LightGBM alvo absoluto"] = pd.Series(m_abs.predict(Xp.loc[te, feats["M2"]]), index=b.index[te])

    # Faixas (alvo secundário): quantis de y_rel no treino
    limiares = y_rel[tr].quantile([0.25, 0.75, 0.95]).to_numpy()
    faixa = lambda r: np.digitize(r, limiares)  # noqa: E731

    dfs = {n: pd.DataFrame({"y": y[te], "p": np.asarray(p, float), "channel_id": b.loc[te, "channel_id"]})
           for n, p in preds.items()}
    boot = bootstrap(dfs)
    linhas = []
    for n, d in dfs.items():
        r = evaluate(d)
        r["f1_faixas"] = f1_score(faixa(y_rel[te]), faixa(d["p"] - base_ch[te]), average="macro")
        r["ic_pairwise"] = ci(boot[n]["pairwise_intra"])
        r["ic_spearman_intra"] = ci(boot[n]["spearman_intra"])
        r["delta_pairwise_vs_B0"] = ci(boot[n]["pairwise_intra"] - boot["B0 mediana do canal"]["pairwise_intra"])
        r["delta_mae_vs_B0"] = ci(boot[n]["mae_log"] - boot["B0 mediana do canal"]["mae_log"])
        linhas.append({"modelo": n, **r})
    tab_temporal = pd.DataFrame(linhas).set_index("modelo")

    # Split por canal ("cliente novo") e CV aleatória (só para medir o otimismo; nunca métrica final)
    n_est = modelos["M2"].best_iteration_
    Xa = Xp.copy()
    canal_k = channel_folds(b["channel_id"])
    rand_k = np.empty(len(b), int)
    for k, (_, idx) in enumerate(KFold(5, shuffle=True, random_state=config.SEED).split(b)):
        rand_k[idx] = k
    outros = {}
    for nome, folds in [("split por canal", canal_k), ("CV aleatória (otimista)", rand_k)]:
        p = oof(Xa, E, y_rel, base_ch, folds, feats["M2"], n_est)
        outros[f"M2 — {nome}"] = evaluate(pd.DataFrame({"y": y, "p": p, "channel_id": b["channel_id"]}))
    outros["B0 — todas as linhas"] = evaluate(pd.DataFrame({"y": y, "p": base_ch, "channel_id": b["channel_id"]}))
    outros["M2 — teste temporal"] = tab_temporal.loc["M2 LightGBM", list(outros["B0 — todas as linhas"])].to_dict()
    tab_outros = pd.DataFrame(outros).T

    imp = pd.Series(modelos["M2"].booster_.feature_importance("gain"), index=feats["M2"])
    imp = (imp / imp.sum()).sort_values(ascending=False)

    # MLflow
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("parallax-modelo-a")
    for nome, m in modelos.items():
        with mlflow.start_run(run_name=nome):
            mlflow.log_params({**PARAMS, "best_iteration": m.best_iteration_, "base": BASE_VERSAO,
                               "cortes": json.dumps(cortes), "n_features": len(feats[nome]), "alvo": "y_rel"})
            mlflow.log_dict({"features": feats[nome]}, "features.json")
            mlflow.log_metrics({k: float(v) for k, v in tab_temporal.loc[f"{nome} LightGBM"].items()
                                if isinstance(v, (int, float, np.floating))})
            if nome == "M2":
                mlflow.log_metrics({f"canal_{k}": float(v) for k, v in outros["M2 — split por canal"].items()})
                mlflow.lightgbm.log_model(m.booster_, name="modelo")
                run_id = mlflow.active_run().info.run_id

    # Artefatos para a inferência (versão de desenvolvimento; v1 é congelada no G2)
    V = config.MODELS / VERSAO_TREINO
    V.mkdir(parents=True, exist_ok=True)
    modelos["M2"].booster_.save_model(V / "ponto.txt")
    for a in config.QUANTIS:
        q[a].booster_.save_model(V / f"q{int(a * 100):02d}.txt")
    joblib.dump(pca, V / "pca_texto.joblib")
    b[["post_id", "title", "channel_id", "sector", "metric_target", "media_ref"]].assign(
        y_rel=y_rel, **{c: Xp[c] for c in PCA_COLS}).to_parquet(V / "comparaveis.parquet", index=False)
    meta = {"versao": VERSAO_TREINO, "base": BASE_VERSAO, "cortes": cortes, "features": feats["M2"], "pca_cols": PCA_COLS,
            "text_emb_model": config.TEXT_EMB_MODEL, "horizonte_dias": config.HORIZONTE_DIAS,
            "limiares_faixa": limiares.tolist(), "largura_p80_val": largura_p80, "mlflow_run": run_id,
            "cobertura_teste_bruta": cobertura, "cobertura_teste_por_setor": cob_setor, "conformal_q": q_conf,
            "cobertura_teste_conformal": cobertura_conf, "cobertura_conformal_por_setor": cob_setor_conf,
            "metricas_teste": tab_temporal.loc["M2 LightGBM", ["mae_log", "spearman_intra", "pairwise_intra"]].to_dict()}
    (V / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    fmt = lambda df: df.to_markdown(floatfmt=".3f")  # noqa: E731
    md = (f"# Resultados G1 — Opção A\n\nBase: {BASE_VERSAO}, canais de treino, posts com ≥{config.MIN_HISTORICO} "
          f"anteriores no canal: {len(b)} linhas.\nCortes temporais: {json.dumps(cortes)}. "
          f"Linhas: treino {tr.sum()}, val {va.sum()}, teste {te.sum()}. MLflow run M2: `{run_id}`.\n\n"
          f"## Teste temporal (IC95% por bootstrap de canais, 300 reamostras)\n\n{fmt(tab_temporal)}\n\n"
          f"## Generalização e otimismo\n\n{fmt(tab_outros)}\n\n"
          f"## Intervalo de 80% (quantis brutos, antes do conformal)\n\nCobertura no teste: {cobertura:.3f} "
          f"(por setor: {cob_setor}). **Com CQR** (q = {q_conf:+.3f}): {cobertura_conf:.3f} "
          f"(por setor: {cob_setor_conf}). Largura P80 no val (log): {largura_p80:.3f}.\n\n"
          f"## Importância (ganho) — M2, top 15\n\n{imp.head(15).to_frame('ganho').to_markdown(floatfmt='.3f')}\n")
    (OUT / f"modelo_{VERSAO_TREINO}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
