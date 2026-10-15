"""F1.2–F1.6: coleta do dia via YouTube Data API v3.

Idempotente e retomável: rodar de novo no mesmo dia só completa os canais que faltam.
Cada dia vira uma pasta nova em data/raw/coleta=AAAA-MM-DD/; coletas anteriores nunca são sobrescritas.

    uv run python scripts/01_collect.py [--budget 9000] [--so-resolver]
"""

import argparse
import json
from datetime import UTC, datetime, timedelta

import pandas as pd

from parallax import config
from parallax.collect.youtube import QuotaExceeded, YouTube, channel_row, video_row

RESOLVED = ["channel_id", "titulo_api", "custom_url", "pais_api", "canal_criado_em", "inscritos", "n_videos",
            "uploads", "motivo"]
MAX_UPLOADS_LISTADOS = 1500


def resolve(yt: YouTube, canais: pd.DataFrame) -> None:
    """Preenche channel_id e estatísticas na planilha de curadoria (handle primeiro, busca por nome se falhar)."""
    for i, r in canais[canais["channel_id"].isna()].iterrows():
        item = yt.channel(handle=r["handle"]) if pd.notna(r["handle"]) else None
        if item is None:
            cid = yt.search_channel_id(r["nome"])
            item = yt.channel(channel_id=cid) if cid else None
        if item is None:
            canais.loc[i, "motivo"] = "canal não encontrado"
            continue
        for k, v in channel_row(item).items():
            canais.loc[i, k] = v
        print(f"  {r['nome']!r} -> {canais.loc[i, 'titulo_api']!r} ({canais.loc[i, 'inscritos']} inscritos)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=9000, help="teto local de unidades de cota nesta execução")
    ap.add_argument("--so-resolver", action="store_true", help="só resolve canais, sem listar vídeos")
    args = ap.parse_args()

    now = datetime.now(UTC)
    out = config.RAW / f"coleta={now.date().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)
    vfile, cfile = out / "videos.parquet", out / "canais.parquet"
    videos = pd.read_parquet(vfile) if vfile.exists() else pd.DataFrame()
    canais_dia = pd.read_parquet(cfile) if cfile.exists() else pd.DataFrame()
    feitos = set(canais_dia["channel_id"]) if len(canais_dia) else set()

    canais = pd.read_csv(config.CANAIS_CSV, dtype=object)  # planilha: tudo como objeto, sem coerção de tipos
    for c in [*RESOLVED, "incluir"]:
        if c not in canais:
            canais[c] = None
    for c in ("inscritos", "n_videos"):
        canais[c] = pd.to_numeric(canais[c])

    yt = YouTube(budget=args.budget)
    since = now - timedelta(days=round(30.44 * config.JANELA_MESES))
    try:
        resolve(yt, canais)
        if args.so_resolver:
            return
        pendentes = canais[canais["channel_id"].notna() & ~canais["channel_id"].isin(feitos)
                           & (canais["incluir"] != "nao")]
        for n, (_, r) in enumerate(pendentes.iterrows(), 1):
            ids = yt.upload_ids(r["uploads"], since, MAX_UPLOADS_LISTADOS)
            batch = pd.DataFrame([video_row(v) for v in yt.videos(ids)])
            if len(batch):
                batch["collected_at"] = now.isoformat()
                batch["inscritos_na_coleta"] = r["inscritos"]
                batch["n_videos_na_coleta"] = r["n_videos"]
                videos = pd.concat([videos, batch]).drop_duplicates("video_id", keep="last")
                videos.to_parquet(vfile, index=False)
            snap = {k: r[k] for k in ["nome", "setor", "tipo", "pais_idioma", *RESOLVED[:-1]]}
            snap.update(collected_at=now.isoformat(), uploads_listados=len(ids), videos_coletados=len(batch))
            canais_dia = pd.concat([canais_dia, pd.DataFrame([snap])])
            canais_dia.to_parquet(cfile, index=False)  # salva por canal: a execução pode parar a qualquer momento
            print(f"[{n}/{len(pendentes)}] {r['titulo_api']}: {len(batch)} vídeos | cota usada {yt.used}")
    except QuotaExceeded as e:
        print(f"Parando com segurança (cota): {e}. Rode de novo após o reset (04h-05h em Brasília).")
    finally:
        canais.to_csv(config.CANAIS_CSV, index=False)
        log = out / "log.jsonl"
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"em": datetime.now(UTC).isoformat(), "unidades": yt.used,
                                "search": yt.search_used, "videos_total": len(videos)}) + "\n")
        print(f"Unidades usadas: {yt.used} (+{yt.search_used} search). Vídeos no snapshot: {len(videos)}")


if __name__ == "__main__":
    main()
