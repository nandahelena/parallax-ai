"""F4.5: onde o modelo erra. Fatias do teste temporal (setor, tamanho do canal, duração, idioma, tipo, histórico)
e os piores casos, com link. Usa os artefatos salvos em models/<versão>.

    uv run python scripts/09_analise_erros.py [--versao v0]
"""

import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from parallax import config
from parallax.model.metrics import evaluate
from parallax.model.train import carregar_treino

OUT = config.ROOT / "docs" / "resultados" / "analise_erros.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versao", default=config.VERSAO_MODELO)
    v = config.MODELS / ap.parse_args().versao
    meta = json.loads((v / "meta.json").read_text(encoding="utf-8"))

    b, X, y, E, split = carregar_treino()
    te = (split == "test").to_numpy()
    X[meta["pca_cols"]] = joblib.load(v / "pca_texto.joblib").transform(E)
    base_ch = X["ch_med_prev20"]
    p = base_ch + lgb.Booster(model_file=str(v / "ponto.txt")).predict(X[meta["features"]])
    d = b.loc[te].assign(y=y[te], p=p[te], erro=(p - y)[te], subs=X.loc[te, "ch_log_subs"].pipe(np.expm1),
                         n_prev=X.loc[te, "ch_n_prev"])

    fatias = {
        "setor": d["sector"],
        "tipo de canal": d["channel_type"],
        "idioma do canal": d["channel_lang"].str[:2].map(lambda s: "BR" if s == "BR" else "global"),
        "inscritos": pd.cut(d["subs"], [0, 1e5, 1e6, 1e7, np.inf], labels=["<100k", "100k-1M", "1M-10M", ">10M"]),
        "duração": pd.cut(d["duration_s"], [0, 15, 30, 60, 180], labels=["≤15s", "15-30s", "30-60s", "60-180s"]),
        "posts anteriores no canal": pd.cut(d["n_prev"], [0, 20, 100, 1000], labels=["3-20", "21-100", ">100"]),
    }
    blocos, achados = [], []
    for nome, chave in fatias.items():
        linhas = []
        for g, sub in d.groupby(chave, observed=True):
            r = evaluate(sub[["y", "p", "channel_id"]])
            linhas.append({nome: g, "n": len(sub), "viés (log)": sub["erro"].mean(), "mae_log": r["mae_log"],
                           "pairwise_intra": r["pairwise_intra"]})
        t = pd.DataFrame(linhas).set_index(nome)
        blocos.append(f"### Por {nome}\n\n{t.to_markdown(floatfmt='.3f')}\n")
        grandes = t[t["n"] >= 50]
        if len(grandes) > 1:
            g = grandes["mae_log"].idxmax()
            achados.append(f"- Por {nome}, o maior erro está em **{g}** (MAE_log {grandes.loc[g, 'mae_log']:.2f} vs "
                           f"{grandes['mae_log'].min():.2f} no melhor grupo; par a par {grandes.loc[g, 'pairwise_intra']:.0%}).")
        pior = grandes["viés (log)"].abs().idxmax() if len(grandes) else None
        if pior is not None and abs(t.loc[pior, "viés (log)"]) > 0.1:
            sentido = "superestima" if t.loc[pior, "viés (log)"] > 0 else "subestima"
            achados.append(f"- O modelo **{sentido}** vídeos com {nome} = **{pior}**: viés médio de "
                           f"{t.loc[pior, 'viés (log)']:+.2f} em log (≈ {np.expm1(abs(t.loc[pior, 'viés (log)'])):.0%} "
                           f"{'a mais' if sentido == 'superestima' else 'a menos'}), n={t.loc[pior, 'n']}.")

    def casos(df):
        return df.assign(previsto=np.expm1(df["p"]).round(), real=df["metric_target"], link=df["media_ref"])[
            ["title", "sector", "channel_type", "previsto", "real", "link"]].assign(title=lambda x: x["title"].str[:60])

    sup = casos(d.nlargest(8, "erro"))
    sub = casos(d.nsmallest(8, "erro"))
    md = (f"# Análise de erros — modelo {meta['versao']} (teste temporal, {te.sum()} Shorts)\n\n"
          f"Erro = previsto − real em escala log1p. Viés > 0: o modelo superestima.\n\n"
          f"## Principais achados (gerados automaticamente)\n\n" + ("\n".join(achados) or "- Nenhuma fatia com viés > 0,15.")
          + "\n\n## Fatias\n\n" + "\n".join(blocos)
          + f"\n## Maiores superestimativas\n\n{sup.to_markdown(index=False)}\n"
          + f"\n## Maiores subestimativas (o modelo não viu o viral chegando)\n\n{sub.to_markdown(index=False)}\n")
    OUT.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
