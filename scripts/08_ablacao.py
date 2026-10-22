"""F3.5–F3.6: ablação da camada multimodal, um bloco por vez, no mesmo subconjunto e na mesma janela (regra 9).

Desenho: o M2 é treinado nos posts de treino FORA do subconjunto multimodal e prevê o subconjunto (offset).
Cada modelo de bloco aprende só o resíduo que o M2 não explica (LightGBM com init_score = offset), treinado
nos vídeos de treino do subconjunto, com early stopping no val, e avaliado no teste. Assim o ganho de cada bloco é
medido sobre um M2 forte, e não sobre um modelo treinado com poucas linhas.
Controle: "M2 + reajuste" (resíduo só com as features do M2) mede o ganho espúrio de reajustar no subconjunto.
Extra: thumbnail em 100% da base (M2 + thumbnail treinado na base inteira).

    uv run python scripts/08_ablacao.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from parallax import config
from parallax.features import media, vlm
from parallax.features.build import cols
from parallax.features.text import encode
from parallax.model.metrics import bootstrap, ci, evaluate
from parallax.model.train import PCA_COLS, add_pca, carregar_treino, fit_lgb

OUT = config.ROOT / "docs" / "resultados" / "ablacao_multimodal.md"
RESIDUO = dict(learning_rate=0.02, num_leaves=7, min_child_samples=60, colsample_bytree=0.3, reg_lambda=10.0)  # forte: poucas linhas com mídia


def main() -> None:
    b, X, y, E, split = carregar_treino()
    base_ch = X["ch_med_prev20"]
    y_rel = y - base_ch
    tr, va, te = ((split == s).to_numpy() for s in ("train", "val", "test"))
    X, _ = add_pca(X, E, tr)
    m2 = cols("canal", "tempo", "metadados", "texto") + PCA_COLS

    # Blocos
    thumb = pd.read_parquet(config.PROCESSED / "emb_thumb.parquet").set_index("post_id").reindex(b["post_id"])
    tem_thumb = thumb.notna().all(axis=1).to_numpy()
    X, _ = add_pca(X, np.nan_to_num(thumb.to_numpy(np.float32)), tr & tem_thumb, "th_pca", 16)
    X.loc[~tem_thumb, [f"th_pca_{i:02d}" for i in range(16)]] = np.nan
    esc, frames = media.carregar(b["post_id"])
    tem_media = esc["vd_dur_s"].notna().to_numpy()
    X = X.join(esc)
    X, pca_frames = add_pca(X, np.nan_to_num(frames), tr & tem_media, "fr_pca", 16)
    X.loc[~tem_media, [f"fr_pca_{i:02d}" for i in range(16)]] = np.nan
    v = vlm.carregar(b["post_id"])
    tem_vlm = (v["vl_falhou"] == 0).to_numpy()
    X = X.join(v.drop(columns="vl_texto"))
    km = KMeans(10, random_state=config.SEED, n_init=10)
    vt = np.zeros((len(b), 384), np.float32)
    if tem_vlm.any():
        vt[tem_vlm] = encode(v.loc[tem_vlm, "vl_texto"], pd.Series([""] * tem_vlm.sum(), index=v.index[tem_vlm]))
        km.fit(vt[tr & tem_vlm] if (tr & tem_vlm).sum() >= 10 else vt[tem_vlm])
        X["vl_cluster"] = np.where(tem_vlm, km.predict(vt), -1)

    blocos = {
        "+ thumbnail": [c for c in X if c.startswith("th_pca")],
        "+ frames (SigLIP2)": [c for c in X if c.startswith("fr_pca")],
        "+ vídeo (OpenCV + cortes)": [c for c in X if c.startswith("vd_")],
        "+ áudio (Whisper + energia)": [c for c in X if c.startswith("au_")],
        "+ detectores (rosto, OCR, objetos)": [c for c in X if c[:3] in ("fc_", "oc_", "ob_")],
        "+ VLM (rubrica + clusters)": [c for c in X if c.startswith("vl_")],
    }

    # Offset: M2 treinado fora do subconjunto, early stopping no val fora do subconjunto
    fora = ~tem_media
    m_off = fit_lgb(X[m2], y_rel, tr & fora, va & fora)
    off = pd.Series(m_off.predict(X[m2]), index=X.index)
    s_tr, s_va, s_te = tr & tem_media, va & tem_media, te & tem_media
    print(f"subconjunto com mídia: treino {s_tr.sum()} | val {s_va.sum()} | teste {s_te.sum()} | VLM {tem_vlm.sum()}")

    preds = {"M2 (offset)": off[s_te]}
    m_ctrl = fit_lgb(X[m2], y_rel, s_tr, s_va, init_score=off.to_numpy(), **RESIDUO)
    preds["M2 + reajuste (controle)"] = off[s_te] + m_ctrl.predict(X.loc[s_te, m2])
    acum = ["sector"]
    residuos = {}
    for nome, fc in blocos.items():
        acum = acum + fc
        m = fit_lgb(X[acum], y_rel, s_tr, s_va, init_score=off.to_numpy(), **RESIDUO)
        preds[nome] = off[s_te] + m.predict(X.loc[s_te, acum])
        residuos[nome] = (m, list(acum))

    dfs = {n: pd.DataFrame({"y": y[s_te], "p": base_ch[s_te] + p, "channel_id": b.loc[s_te, "channel_id"]})
           for n, p in preds.items()}
    boot = bootstrap(dfs)
    ref = boot["M2 (offset)"]
    linhas = []
    for n, d in dfs.items():
        r = evaluate(d)
        linhas.append({"modelo (cumulativo)": n, "mae_log": r["mae_log"], "spearman_intra": r["spearman_intra"],
                       "pairwise_intra": r["pairwise_intra"],
                       "Δ pairwise vs M2 (IC95%)": ci(boot[n]["pairwise_intra"] - ref["pairwise_intra"]),
                       "Δ spearman_intra vs M2 (IC95%)": ci(boot[n]["spearman_intra"] - ref["spearman_intra"])})
    tab = pd.DataFrame(linhas).set_index("modelo (cumulativo)")

    # Extra: thumbnail na base inteira (100% das linhas têm thumbnail)
    fc_th = m2 + blocos["+ thumbnail"]
    mt = {n: fit_lgb(X[f], y_rel, tr, va) for n, f in [("M2", m2), ("M2 + thumbnail", fc_th)]}
    dfs_th = {n: pd.DataFrame({"y": y[te], "p": base_ch[te] + m.predict(X.loc[te, f]), "channel_id": b.loc[te, "channel_id"]})
              for (n, m), f in zip(mt.items(), [m2, fc_th])}
    bt = bootstrap(dfs_th)
    th = pd.DataFrame({n: evaluate(d) for n, d in dfs_th.items()}).T[["mae_log", "spearman_intra", "pairwise_intra"]]
    th["Δ pairwise vs M2 (IC95%)"] = [ci(bt[n]["pairwise_intra"] - bt["M2"]["pairwise_intra"]) for n in th.index]

    md = (f"# Ablação multimodal\n\nSubconjunto com mídia extraída: treino {s_tr.sum()}, val {s_va.sum()}, "
          f"teste {s_te.sum()} (de {te.sum()} no teste); VLM em {tem_vlm.sum()} vídeos.\n\n"
          f"## Blocos sobre o M2 (resíduo, cumulativo)\n\n{tab.to_markdown(floatfmt='.3f')}\n\n"
          f"## Thumbnail na base inteira\n\n{th.to_markdown(floatfmt='.3f')}\n")
    # Integração no modelo final: só se algum bloco tiver ganho significativo (IC95% do Δ par a par > 0)
    v = config.MODELS / config.VERSAO_MODELO
    for arq in ("residuo_midia.txt", "residuo_midia.json", "pca_frames.joblib"):
        (v / arq).unlink(missing_ok=True)
    delta = {n: boot[n]["pairwise_intra"] - ref["pairwise_intra"] for n in residuos}
    sig = {n: d.mean() for n, d in delta.items() if d.quantile(0.025) > 0}
    if sig:
        melhor = max(sig, key=sig.get)
        m, fc = residuos[melhor]
        m.booster_.save_model(v / "residuo_midia.txt")
        (v / "residuo_midia.json").write_text(json.dumps({"bloco_ate": melhor, "features": fc,
            "delta_pairwise": round(float(sig[melhor]), 4)}, ensure_ascii=False, indent=2), encoding="utf-8")
        joblib.dump(pca_frames, v / "pca_frames.joblib")
        md += f"\n**Integrado ao {config.VERSAO_MODELO}:** resíduo até `{melhor}` (Δ par a par {sig[melhor]:+.3f}).\n"
    else:
        md += (f"\n**Não integrado:** nenhum bloco com IC95% do Δ par a par acima de zero; o {config.VERSAO_MODELO} "
               "usa canal + tempo + texto.\n")
    OUT.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
