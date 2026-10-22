"""F7.1: demonstração "de-para" nos 9 canais guardados (nunca vistos no treino).

Para cada canal, os 4 Shorts mais recentes viram 4 criativos candidatos "ainda não publicados": o histórico enviado
à tool (via `channel_stats`, caminho agnóstico de cliente novo) só tem posts anteriores ao primeiro deles.
O agente prevê e ordena; comparamos com as curtidas reais. Respostas salvas em demo/respostas/ (plano B).

    uv run python scripts/10_demo.py [--narrar]
"""

import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd

from parallax import config
from parallax.agent.graph import executar
from parallax.schemas import ChannelStats, PostAnterior, PredictionRequest

OUT_JSON = config.ROOT / "demo" / "respostas"
OUT_MD = config.ROOT / "docs" / "resultados" / "demo.md"
N_CRIATIVOS = 4


def pedidos_do_canal(g: pd.DataFrame, video: bool = False) -> tuple[list[PredictionRequest], pd.DataFrame]:
    g = g.sort_values("published_at")
    alvo, hist = g.tail(N_CRIATIVOS), g.iloc[:-N_CRIATIVOS].tail(60)
    r0 = g.iloc[0]
    stats = ChannelStats(
        subscribers=int(r0["channel_stats_at_collection"]) if pd.notna(r0["channel_stats_at_collection"]) else None,
        video_count=int(r0["channel_videos_at_collection"]) if pd.notna(r0["channel_videos_at_collection"]) else None,
        is_brand=r0["channel_type"] == "marca", language="pt" if str(r0["channel_lang"]).startswith("BR") else "en",
        recent_posts=[PostAnterior(published_at=t, likes=int(l)) for t, l in zip(hist["published_at"], hist["metric_target"])])
    reqs = [PredictionRequest(creative=r.media_ref, caption=f"{r.title}\n{r.description}",
                              planned_datetime=r.published_at.to_pydatetime(), sector=r.sector, channel_stats=stats,
                              analisar_video=video)
            for r in alvo.itertuples()]
    return reqs, alvo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrar", action="store_true", help="explicação pelo subagente LLM (gasta cota do Gemini)")
    ap.add_argument("--video", action="store_true", help="baixar e analisar cada vídeo (~20 s por criativo)")
    args = ap.parse_args()
    narrar = args.narrar
    b = pd.read_parquet(config.PROCESSED / config.BASE_TREINO)
    demo = b[b["role"] == "demo"]
    nomes = pd.read_csv(config.CANAIS_CSV).dropna(subset=["channel_id"]).set_index("channel_id")["titulo_api"].to_dict()
    OUT_JSON.mkdir(parents=True, exist_ok=True)

    linhas, secoes = [], []
    for cid, g in demo.groupby("channel_id"):
        reqs, alvo = pedidos_do_canal(g, args.video)
        estado = executar(reqs, narrar)
        nome = nomes.get(cid, cid)
        rk = estado["ranking"]
        (OUT_JSON / f"{cid}.json").write_text(rk.model_dump_json(indent=2), encoding="utf-8")
        real = alvo["metric_target"].to_numpy()
        prev = np.array([r.predicted_metric for r in estado["respostas"]])
        lo = np.array([r.interval_low for r in estado["respostas"]])
        hi = np.array([r.interval_high for r in estado["respostas"]])
        pares = [(i, j) for i, j in combinations(range(len(real)), 2) if real[i] != real[j]]
        acertos = sum((prev[i] - prev[j]) * (real[i] - real[j]) > 0 for i, j in pares)
        ordem_real = list(np.argsort(-real))
        top_ok = rk.ranking[0].indice == ordem_real[0]
        for k, (r, a) in enumerate(zip(estado["respostas"], alvo.itertuples())):
            linhas.append({"canal": nome, "setor": a.sector, "titulo": a.title[:50], "previsto": r.predicted_metric,
                           "intervalo": f"{r.interval_low:.0f}–{r.interval_high:.0f}", "real": int(a.metric_target),
                           "dentro": bool(r.interval_low <= a.metric_target <= r.interval_high),
                           "erro_log": abs(np.log1p(r.predicted_metric) - np.log1p(a.metric_target))})
        tab = pd.DataFrame(linhas[-len(reqs):]).drop(columns=["canal", "setor"])
        tab.insert(0, "rank previsto", [next(i.posicao for i in rk.ranking if i.indice == k) for k in range(len(reqs))])
        tab.insert(1, "rank real", [ordem_real.index(k) + 1 for k in range(len(reqs))])
        secoes.append(f"### {nome} ({alvo['sector'].iloc[0]}) · trilha do agente: {' → '.join(estado['trilha'])}\n\n"
                      f"Pares ordenados certo: **{acertos}/{len(pares)}** · 1º lugar previsto é o 1º real: "
                      f"**{'sim' if top_ok else 'não'}** · P(1º > 2º) = {rk.prob_primeiro_supera_segundo}\n\n"
                      f"{tab.to_markdown(index=False, floatfmt='.2f')}\n\n"
                      f"> Explicação do 1º colocado: {estado['respostas'][rk.ranking[0].indice].explanation}\n")
        print(f"{nome}: {acertos}/{len(pares)} pares certos", flush=True)

    d = pd.DataFrame(linhas)
    resumo = {"criativos": len(d), "canais": d["canal"].nunique(), "cobertura_intervalo": round(d["dentro"].mean(), 3),
              "mae_log": round(d["erro_log"].mean(), 3)}
    pares_tot = acertos_tot = 0
    for _, g in d.groupby("canal"):
        p, r = g["previsto"].to_numpy(), g["real"].to_numpy()
        for i, j in combinations(range(len(g)), 2):
            if r[i] != r[j]:
                pares_tot += 1
                acertos_tot += (p[i] - p[j]) * (r[i] - r[j]) > 0
    resumo["acerto_par_a_par"] = round(acertos_tot / pares_tot, 3)
    (OUT_JSON / "_resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    md = (f"# Demonstração de-para — canais nunca vistos no treino\n\nCada canal: os {N_CRIATIVOS} Shorts mais recentes "
          f"tratados como candidatos não publicados; histórico enviado via `channel_stats` só com posts anteriores.\n\n"
          f"**Resumo:** {resumo['criativos']} criativos de {resumo['canais']} canais · acerto par a par "
          f"**{resumo['acerto_par_a_par']:.0%}** · cobertura do intervalo de 80%: **{resumo['cobertura_intervalo']:.0%}** "
          f"· MAE_log {resumo['mae_log']}\n\nRespostas completas da tool em `demo/respostas/` (plano B da apresentação).\n\n"
          + "\n".join(secoes))
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
