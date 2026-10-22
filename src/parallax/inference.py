"""API principal (D16): `prever_criativo(req) -> PredictionResponse`. O LangGraph e o MCP só a embrulham.

O número vem sempre do LightGBM; o texto da explicação só reapresenta números do payload (PLANO regra 1).
"""

import json
import re
from datetime import timedelta
from functools import cache
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from parallax import config
from parallax.features.build import ROTULOS, tabular_features
from parallax.features.text import encode
from parallax.schemas import (Comparable, Driver, PredictionRequest, PredictionResponse, RankItem, RankResponse,
                              WhatIfItem)

VERSAO = config.VERSAO_MODELO
CANAL_NOVO = "__canal__"


@cache
def artefatos() -> dict:
    v = config.MODELS / VERSAO
    if not (v / "meta.json").exists():
        raise FileNotFoundError(f"{v} sem artefatos: rode scripts/07_treinar.py")
    return {
        "meta": json.loads((v / "meta.json").read_text(encoding="utf-8")),
        "ponto": lgb.Booster(model_file=str(v / "ponto.txt")),
        **{f"q{int(a * 100):02d}": lgb.Booster(model_file=str(v / f"q{int(a * 100):02d}.txt"))
           for a in config.QUANTIS},
        "pca": joblib.load(v / "pca_texto.joblib"),
        "comp": pd.read_parquet(v / "comparaveis.parquet"),
        # camada multimodal (opcional): resíduo sobre o modelo base, treinado na ablação (08_ablacao.py)
        **({"residuo": lgb.Booster(model_file=str(v / "residuo_midia.txt")),
            "residuo_meta": json.loads((v / "residuo_midia.json").read_text(encoding="utf-8")),
            "pca_frames": joblib.load(v / "pca_frames.joblib")} if (v / "residuo_midia.txt").exists() else {}),
    }


def _historico_api(channel_id: str) -> tuple[list[tuple], dict]:
    """Shorts do canal via YouTube Data API (~10 unidades de cota)."""
    from datetime import UTC, datetime

    from parallax.collect.youtube import YouTube, channel_row, video_row
    from parallax.data.dataset import shorts_mask

    yt = YouTube(budget=50)
    item = yt.channel(channel_id=channel_id)
    if item is None:
        raise ValueError(f"canal {channel_id} não encontrado")
    ch = channel_row(item)
    v = pd.DataFrame([video_row(x) for x in yt.videos(yt.upload_ids(ch["uploads"], datetime.now(UTC) - timedelta(days=365), 200))])
    v = v[shorts_mask(v)] if len(v) else v
    agora = pd.Timestamp.now(tz="UTC")
    posts = [(pd.Timestamp(r.published_at),
              # likes de posts com < 30 dias ainda não estabilizaram: entram só na cadência, não no patamar
              r.like_count if (agora - pd.Timestamp(r.published_at)).days >= config.IDADE_MIN_DIAS else None)
             for r in v.itertuples()]
    return posts, {"subs": ch["inscritos"], "videos": ch["n_videos"], "marca": True,  # clientes Galaxies: marcas
                   "lang_br": ch["pais_api"] == "BR"}


def _duracao(creative: str, warnings: list[str]) -> float | None:
    if m := re.search(r"(?:shorts/|v=|youtu\.be/)([\w-]{11})", creative):
        from parallax.collect.youtube import YouTube, video_row

        vids = YouTube(budget=5).videos([m.group(1)])
        return video_row(vids[0])["duration_s"] if vids else None
    if Path(creative).suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"} and Path(creative).exists():
        import cv2

        cap = cv2.VideoCapture(creative)
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return n / fps if fps else None
    warnings.append("duração do vídeo desconhecida (criativo em texto ou imagem)")
    return None


def montar_features(req: PredictionRequest) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Linha de features do criativo pelo MESMO código do treino (histórico + post novo na última linha)."""
    warnings = []
    if req.channel_stats:
        cs = req.channel_stats
        posts = [(pd.Timestamp(p.published_at), p.likes) for p in cs.recent_posts]
        info = {"subs": cs.subscribers, "videos": cs.video_count, "marca": cs.is_brand,
                "lang_br": cs.language.lower().startswith("pt")}
    else:
        posts, info = _historico_api(req.channel_id)
    quando = pd.Timestamp(req.planned_datetime)
    quando = quando.tz_localize(config.TZ_LOCAL) if quando.tzinfo is None else quando
    quando = quando.tz_convert("UTC")
    posts = [(t if t.tzinfo else t.tz_localize("UTC"), l) for t, l in posts if (t if t.tzinfo else t.tz_localize("UTC")) < quando]

    titulo, _, descricao = req.caption.partition("\n")
    linhas = [{"published_at": t, "metric_target": l, "title": "", "description": ""} for t, l in posts]
    linhas.append({"published_at": quando, "metric_target": None, "title": titulo, "description": descricao})
    df = pd.DataFrame(linhas).assign(
        channel_id=CANAL_NOVO, sector=req.sector, channel_stats_at_collection=info["subs"],
        channel_videos_at_collection=info["videos"], channel_type="marca" if info["marca"] else "criador",
        channel_lang="BR/pt" if info["lang_br"] else "XX/en", duration_s=_duracao(req.creative, warnings),
        collected_at=quando + timedelta(days=config.HORIZONTE_DIAS))  # idade de controle = horizonte H
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["metric_target"] = pd.to_numeric(df["metric_target"])
    df["duration_s"] = pd.to_numeric(df["duration_s"])  # None (duração desconhecida) → NaN, que o GBM aceita
    a = artefatos()
    x = tabular_features(df).iloc[[-1]].reset_index(drop=True)
    z = a["pca"].transform(encode(pd.Series([titulo]), pd.Series([descricao])))
    x[a["meta"]["pca_cols"]] = z
    return x[a["meta"]["features"]], z[0], warnings


# ---------- as 5 tools do TAPI (o grafo LangGraph chama exatamente estas) ----------
def extrair_atributos(req: PredictionRequest) -> dict:
    """Tool 1: atributos do criativo e do contexto (mesmo código de features do treino)."""
    x, z, warnings = montar_features(req)
    if pd.isna(x["ch_med_prev20"].iloc[0]):
        raise ValueError("o canal precisa de pelo menos 1 post anterior com curtidas maduras (≥30 dias)")
    usados = {k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v)))
              for k, v in x.iloc[0].items() if not k.startswith("tx_pca")}
    x_midia = None
    midia = None
    if req.analisar_video:
        from parallax.features.media import atributos_do_criativo
        try:
            midia = atributos_do_criativo(req.creative)
        except Exception as e:  # noqa: BLE001 - vídeo inacessível não impede a previsão por canal + texto
            warnings.append(f"não foi possível processar o vídeo ({str(e)[:80]}); previsão usa canal, tempo e texto")
    if midia is None:
        warnings.append("criativo sem vídeo processado: previsão usa canal, tempo e texto")
    else:
        f, emb, txt = midia
        usados |= {f"midia_{k}": float(v) for k, v in f.items() if isinstance(v, (int, float)) and not pd.isna(v)}
        usados["midia_transcricao"] = txt[:200]
        a = artefatos()
        if "residuo" in a:
            pcs = a["pca_frames"].transform(emb.reshape(1, -1))[0]
            linha = {**f, **{f"fr_pca_{i:02d}": v for i, v in enumerate(pcs)}, "sector": x["sector"].iloc[0]}
            x_midia = pd.DataFrame([linha]).reindex(columns=a["residuo_meta"]["features"])
            x_midia["sector"] = x_midia["sector"].astype(x["sector"].dtype)
        else:
            warnings.append(f"atributos de vídeo extraídos para inspeção; o modelo {VERSAO} não os usa no número")
    return {"x": x, "z": z, "x_midia": x_midia, "warnings": warnings, "features_used": usados}


def comparaveis(sector: str, z: np.ndarray, k: int = 5, por_canal: int = 2) -> list[Comparable]:
    """Tool 2: Shorts reais do mesmo setor mais parecidos com o criativo (no máximo `por_canal` por canal)."""
    comp = artefatos()["comp"]
    comp = comp[comp["sector"] == sector]
    m = comp[artefatos()["meta"]["pca_cols"]].to_numpy()
    sim = pd.Series(m @ z / (np.linalg.norm(m, axis=1) * np.linalg.norm(z) + 1e-9), index=comp.index)
    top = comp.assign(sim=sim).sort_values("sim", ascending=False).groupby("channel_id").head(por_canal).head(k)
    return [Comparable(post_id=r.post_id, url=r.media_ref, title=r.title, likes=int(r.metric_target),
                       relative_to_channel=round(float(np.exp(r.y_rel)), 2), similarity=round(float(r.sim), 3))
            for r in top.itertuples()]


def prever(x: pd.DataFrame, x_midia: pd.DataFrame | None = None) -> dict:
    """Tool 3: o número. Ponto e intervalo de 80% vêm do LightGBM (nunca do LLM)."""
    a = artefatos()
    rel = float(a["ponto"].predict(x)[0]) + residuo_midia(x_midia)
    q = a["meta"].get("conformal_q", 0.0)  # CQR calibrado no treino (0 em versões sem calibração)
    # q < 0 estreita o intervalo: reordenar depois do ajuste garante low ≤ pred ≤ high
    lo, hi = min(float(a["q10"].predict(x)[0]) - q, rel), max(float(a["q90"].predict(x)[0]) + q, rel)
    base_ch = float(x["ch_med_prev20"].iloc[0])
    pred, low, high = (float(np.expm1(base_ch + v)) for v in (rel, lo, hi))
    motivos = []
    if x["ch_n_prev"].iloc[0] < 5:
        motivos.append("histórico curto do canal (menos de 5 posts anteriores)")
    if hi - lo > a["meta"]["largura_p80_val"]:
        motivos.append("intervalo mais largo que 80% das previsões de validação")
    return {"predicted_metric": round(pred, 1), "interval_low": round(low, 1), "interval_high": round(high, 1),
            "relative_to_channel": round(float(np.exp(rel)), 2), "confidence_flag": bool(motivos),
            "confidence_reasons": motivos}


def residuo_midia(x_midia: pd.DataFrame | None) -> float:
    """Ajuste do modelo de vídeo/áudio (resíduo aprendido sobre o modelo de texto+histórico na ablação).

    shortcut: o resíduo foi treinado sobre o offset do M2 da ablação, não sobre o ponto final do v1 (mesmas
    features); a diferença é pequena. Retreinar o resíduo sobre o v1 se a camada multimodal virar central.
    """
    a = artefatos()
    return float(a["residuo"].predict(x_midia)[0]) if x_midia is not None and "residuo" in a else 0.0


def explicar(x: pd.DataFrame, n: int = 5, x_midia: pd.DataFrame | None = None) -> list[Driver]:
    """Tool 4: fatores que mais pesaram (TreeSHAP nativo do LightGBM, componentes de texto agrupados)."""
    contrib = pd.Series(artefatos()["ponto"].predict(x, pred_contrib=True)[0][:-1], index=x.columns)
    contrib = contrib.groupby(lambda c: "tx_pca" if c.startswith("tx_pca") else c).sum()
    if x_midia is not None and "residuo" in artefatos():
        contrib["midia"] = residuo_midia(x_midia)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(n)
    return [Driver(factor=ROTULOS.get(k, k), direction="aumenta" if v > 0 else "reduz", weight=round(float(v), 3))
            for k, v in top.items()]


def explicacao_template(p: dict, drivers: list[Driver]) -> str:
    br = lambda v: f"{v:,.0f}".replace(",", ".")  # noqa: E731
    fatores = "; ".join(f"{d.factor} ({d.direction})" for d in drivers[:3])
    return (f"Previsão de ~{br(p['predicted_metric'])} curtidas em {artefatos()['meta']['horizonte_dias']} dias "
            f"(intervalo de 80%: {br(p['interval_low'])} a {br(p['interval_high'])}), "
            f"{str(p['relative_to_channel']).replace('.', ',')}× a mediana recente do canal. "
            f"Fatores que mais pesaram: {fatores}.")


def montar_resposta(at: dict, p: dict, drivers: list[Driver], comps: list[Comparable],
                    narrar: bool = False) -> PredictionResponse:
    resp = PredictionResponse(
        **p, top_drivers=drivers, explanation=explicacao_template(p, drivers), comparables=comps,
        features_used=at["features_used"], model_version=f"parallax-{VERSAO} ({artefatos()['meta']['base']})",
        warnings=at["warnings"])
    if narrar:  # subagente explicador (LLM); só substitui o template se passar no validador
        from parallax.explain.narrator import narrar as _narrar
        texto, status = _narrar(resp)
        if texto:
            resp.explanation = texto
        resp.warnings.append(status)
    return resp


def prever_criativo(req: PredictionRequest, narrar: bool = False) -> PredictionResponse:
    """Prevê as curtidas de um Short antes da publicação, com intervalo de 80%, fatores e comparáveis.

    Entrada: criativo, legenda planejada, data/hora planejada, setor e o canal (`channel_id` para buscar o histórico
    na YouTube API, ou `channel_stats` com inscritos e posts anteriores, o que dispensa a nossa base).
    `narrar=True` pede ao subagente LLM uma explicação em linguagem natural, validada contra o JSON.
    """
    at = extrair_atributos(req)
    return montar_resposta(at, prever(at["x"], at["x_midia"]), explicar(at["x"], x_midia=at["x_midia"]),
                           comparaveis(req.sector, at["z"]), narrar)


def ordenar(previsoes: list[PredictionResponse], creatives: list[str]) -> RankResponse:
    """Tool 5: ranking por curtidas previstas e P(1º > 2º) pela incerteza dos intervalos.

    shortcut: trata os erros dos dois criativos como independentes; no mesmo canal eles se correlacionam, então a
    probabilidade sai conservadora. Usar a covariância dos resíduos intra-canal se virar decisão crítica.
    """
    from scipy.stats import norm

    ordem = sorted(range(len(previsoes)), key=lambda i: -previsoes[i].predicted_metric)
    itens = [RankItem(posicao=k + 1, indice=i, creative=creatives[i], predicted_metric=previsoes[i].predicted_metric,
                      interval_low=previsoes[i].interval_low, interval_high=previsoes[i].interval_high,
                      relative_to_channel=previsoes[i].relative_to_channel) for k, i in enumerate(ordem)]
    prob = None
    if len(ordem) > 1:
        a, b = (previsoes[i] for i in ordem[:2])
        sig = lambda r: (np.log1p(r.interval_high) - np.log1p(r.interval_low)) / 2.563  # noqa: E731 (80% = ±1,28σ)
        z = (np.log1p(a.predicted_metric) - np.log1p(b.predicted_metric)) / max(np.hypot(sig(a), sig(b)), 1e-9)
        prob = round(float(norm.cdf(z)), 3)
    return RankResponse(ranking=itens, prob_primeiro_supera_segundo=prob, previsoes=previsoes)


def ranquear_criativos(reqs: list[PredictionRequest], narrar: bool = False) -> RankResponse:
    """Prevê vários criativos (ex.: candidatos da mesma campanha) e os ordena do mais ao menos promissor."""
    return ordenar([prever_criativo(r, narrar) for r in reqs], [r.creative for r in reqs])


def simular_what_if(req: PredictionRequest, variacoes: list[dict[str, str]]) -> list[WhatIfItem]:
    """Re-pontua o criativo trocando a legenda e/ou a data planejada (chaves `caption`, `planned_datetime`)."""
    base = prever_criativo(req).predicted_metric
    out = []
    for v in variacoes:
        alt = PredictionRequest.model_validate({**req.model_dump(), **v})
        p = prever_criativo(alt).predicted_metric
        out.append(WhatIfItem(variacao=v, predicted_metric=p, delta_vs_original_pct=round(100 * (p / base - 1), 1)))
    return out
