"""F4.6 / R29: recoleta das estatísticas antes do treino final (guia da liga + política de dados da API).

Novo snapshot em data/raw/coleta=<hoje>/estatisticas.parquet (a coleta original nunca é sobrescrita) e
data/processed/base_v2.parquet = base_v1 com curtidas, inscritos e data de coleta atualizados.
Vídeos que sumiram ou ocultaram as curtidas saem (e são contados).

    uv run python scripts/03_recoletar_metricas.py
"""

import json
from datetime import UTC, datetime

import pandas as pd

from parallax import config
from parallax.collect.youtube import YouTube


def main() -> None:
    b = pd.read_parquet(config.PROCESSED / "base_v1.parquet")
    agora = datetime.now(UTC)
    out = config.RAW / f"coleta={agora.date().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)
    yt = YouTube(budget=1500)

    ids = b["post_id"].tolist()
    stats = []
    for i in range(0, len(ids), 50):
        resp = yt._exec(yt.api.videos().list(part="statistics", id=",".join(ids[i:i + 50])))
        stats += [{"post_id": it["id"], "like_count": int(it["statistics"]["likeCount"])
                   if "likeCount" in it["statistics"] else None} for it in resp.get("items", [])]
    canais = []
    cids = b["channel_id"].unique().tolist()
    for i in range(0, len(cids), 50):
        resp = yt._exec(yt.api.channels().list(part="statistics", id=",".join(cids[i:i + 50])))
        canais += [{"channel_id": it["id"], "subs": int(it["statistics"]["subscriberCount"])
                    if "subscriberCount" in it["statistics"] else None,
                    "videos": int(it["statistics"].get("videoCount", 0))} for it in resp.get("items", [])]
    s, c = pd.DataFrame(stats), pd.DataFrame(canais)
    s.assign(collected_at=agora.isoformat()).to_parquet(out / "estatisticas.parquet", index=False)
    c.assign(collected_at=agora.isoformat()).to_parquet(out / "canais_estatisticas.parquet", index=False)

    v2 = b.drop(columns=["metric_target", "channel_stats_at_collection", "channel_videos_at_collection", "collected_at"])
    v2 = v2.merge(s.rename(columns={"like_count": "metric_target"}), on="post_id", how="left")
    v2 = v2.merge(c.rename(columns={"subs": "channel_stats_at_collection", "videos": "channel_videos_at_collection"}),
                  on="channel_id", how="left")
    v2["collected_at"] = pd.Timestamp(agora)
    funil = {"base_v1": len(b), "sumiram (removidos/privados)": int((~b["post_id"].isin(s["post_id"])).sum()),
             "curtidas ocultas agora": int(v2["metric_target"].isna().sum() - (~b["post_id"].isin(s["post_id"])).sum())}
    v2 = v2[v2["metric_target"].notna()].copy()
    v2["metric_target"] = v2["metric_target"].astype("int64")
    funil["base_v2"] = len(v2)
    variacao = (v2.set_index("post_id")["metric_target"] / b.set_index("post_id")["metric_target"]).dropna()
    funil["variacao_curtidas_mediana"] = round(float(variacao.median()), 4)
    funil["unidades_de_cota"] = yt.used
    v2.to_parquet(config.PROCESSED / "base_v2.parquet", index=False)
    (config.PROCESSED / "funil_v2.json").write_text(json.dumps(funil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(funil, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
