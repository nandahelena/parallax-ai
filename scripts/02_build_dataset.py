"""F1.5–F1.6: snapshot bruto → data/processed/base_v1.parquet (esquema mínimo do TAPI + extras) e funil.

Também fecha a curadoria: marca incluir/motivo por canal e sorteia (uma vez só) os canais de demo.

    uv run python scripts/02_build_dataset.py [--coleta AAAA-MM-DD]
"""

import argparse
import json

import numpy as np
import pandas as pd

from parallax import config
from parallax.data.dataset import shorts_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coleta", help="data da coleta (padrão: a mais recente)")
    args = ap.parse_args()
    pasta = (config.RAW / f"coleta={args.coleta}") if args.coleta else sorted(config.RAW.glob("coleta=*"))[-1]

    v = pd.read_parquet(pasta / "videos.parquet")
    canais = pd.read_csv(config.CANAIS_CSV, dtype=object)
    # Um canal pode aparecer em duas linhas (ex.: duplicata marcada "nao"); só sai se nenhuma linha o mantiver.
    excluidos_manual = (set(canais.loc[canais["incluir"] == "nao", "channel_id"].dropna())
                        - set(canais.loc[canais["incluir"] != "nao", "channel_id"].dropna()))

    funil = {"coletados": len(v)}

    def etapa(nome: str, manter: pd.Series) -> None:
        nonlocal v
        v = v[manter.to_numpy()]
        funil[nome] = len(v)

    etapa("sem canais excluidos na curadoria", ~v["channel_id"].isin(excluidos_manual))
    etapa("sem live/premiere", ~v["was_live"].astype(bool) & (v["live_broadcast_content"].fillna("none") == "none"))
    funil["orientacao desconhecida (excluidos)"] = int(v["embed_h"].isna().sum())
    etapa("heuristica de Shorts", shorts_mask(v))
    idade = (pd.to_datetime(v["collected_at"], utc=True) - pd.to_datetime(v["published_at"], utc=True)).dt.days
    etapa(f"idade >= {config.IDADE_MIN_DIAS} dias", idade >= config.IDADE_MIN_DIAS)
    etapa("likes visiveis (ocultos = NaN, nunca zero)", v["like_count"].notna())
    etapa("sem duplicatas", ~v["video_id"].duplicated())
    v = v.sort_values("published_at", ascending=False)
    etapa(f"no maximo {config.MAX_SHORTS_POR_CANAL} por canal",
          v.groupby("channel_id").cumcount() < config.MAX_SHORTS_POR_CANAL)

    # Curadoria final: canal entra se tiver Shorts elegíveis suficientes.
    n_por_canal = v.groupby("channel_id").size()
    canais["shorts_elegiveis"] = canais["channel_id"].map(n_por_canal).fillna(0).astype(int)
    auto = canais["incluir"] != "nao"
    poucos = auto & (canais["shorts_elegiveis"] < config.MIN_SHORTS_POR_CANAL)
    canais.loc[poucos, "motivo"] = "apenas " + canais.loc[poucos, "shorts_elegiveis"].astype(str) + " Shorts elegiveis"
    canais.loc[auto, "incluir"] = np.where(poucos[auto], "nao", "sim")
    etapa(f"canais com >= {config.MIN_SHORTS_POR_CANAL} Shorts",
          v["channel_id"].isin(canais.loc[canais["incluir"] == "sim", "channel_id"]))

    # Demo: sorteio estratificado por setor, fixado na planilha para nunca mudar entre execuções.
    if "papel" not in canais or canais["papel"].isna().all():
        canais["papel"] = None
        rng = np.random.default_rng(config.SEED)
        for setor, g in canais[canais["incluir"] == "sim"].groupby("setor"):
            demo = rng.choice(g.index.to_numpy(), size=min(config.N_DEMO_POR_SETOR, len(g)), replace=False)
            canais.loc[g.index, "papel"] = "treino"
            canais.loc[demo, "papel"] = "demo"
    canais.to_csv(config.CANAIS_CSV, index=False)

    meta = canais.loc[canais["incluir"] == "sim"].drop_duplicates("channel_id").set_index("channel_id")
    base = pd.DataFrame({
        # esquema mínimo do TAPI
        "post_id": v["video_id"],
        "channel_id": v["channel_id"],
        "published_at": pd.to_datetime(v["published_at"], utc=True),
        "caption": v["title"] + "\n" + v["description"],
        "media_ref": "https://www.youtube.com/shorts/" + v["video_id"],
        "metric_target": v["like_count"].astype("int64"),
        "channel_stats_at_collection": v["inscritos_na_coleta"],
        "sector": v["channel_id"].map(meta["setor"]),
        # extras
        "title": v["title"],
        "description": v["description"],
        "duration_s": v["duration_s"],
        "thumbnail_url": v["thumbnail_url"],
        "collected_at": pd.to_datetime(v["collected_at"], utc=True),
        "channel_videos_at_collection": v["n_videos_na_coleta"],
        "channel_type": v["channel_id"].map(meta["tipo"]),
        "channel_lang": v["channel_id"].map(meta["pais_idioma"]),
        "role": v["channel_id"].map(meta["papel"]),
        # só para análise: métricas pós-publicação NUNCA viram feature (PLANO.md regra 3)
        "view_count_pos": v["view_count"],
        "comment_count_pos": v["comment_count"],
    })
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    base.to_parquet(config.PROCESSED / "base_v1.parquet", index=False)

    funil["canais incluidos"] = int((canais["incluir"] == "sim").sum())
    funil["por setor"] = base.groupby("sector").agg(videos=("post_id", "size"),
                                                     canais=("channel_id", "nunique")).to_dict("index")
    funil["papel"] = base.groupby("role").agg(videos=("post_id", "size"),
                                               canais=("channel_id", "nunique")).to_dict("index")
    funil["periodo"] = [str(base["published_at"].min().date()), str(base["published_at"].max().date())]
    funil["coleta"] = pasta.name
    (config.PROCESSED / "funil_v1.json").write_text(json.dumps(funil, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(json.dumps(funil, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
