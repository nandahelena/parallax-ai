"""F1.7 + F3.3: thumbnails de 100% da base (CDN do YouTube, sem cota da API) e embedding SigLIP2 de cada uma.
Retomável: pula thumbnails já baixadas e embeddings já no cache.

    uv run python scripts/04_extrair_thumbnails.py
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import httpx
import numpy as np
import pandas as pd
import torch

from parallax import config
from parallax.features.media import image_embeddings

CACHE = config.PROCESSED / "emb_thumb.parquet"


def baixar(row) -> None:
    destino = config.MEDIA / row.post_id / "thumb.jpg"
    if destino.exists():
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = httpx.get(row.thumbnail_url, timeout=30, follow_redirects=True)
    except httpx.TransportError:  # falha de rede: fica para a próxima execução (retomável)
        return
    if r.status_code == 200:
        destino.write_bytes(r.content)


def main() -> None:
    torch.set_num_threads(4)  # divide a CPU com a extração de mídia que roda em paralelo
    b = pd.read_parquet(config.PROCESSED / "base_v1.parquet")
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(baixar, b.itertuples()))
    print("thumbnails ok", flush=True)

    feito = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["post_id"])
    falta = [v for v in b["post_id"] if v not in set(feito["post_id"]) and (config.MEDIA / v / "thumb.jpg").exists()]
    for i in range(0, len(falta), 256):  # grava a cada lote: retomável
        ids = falta[i:i + 256]
        imgs = [cv2.cvtColor(cv2.imread(str(config.MEDIA / v / "thumb.jpg")), cv2.COLOR_BGR2RGB) for v in ids]
        emb = pd.DataFrame(np.vstack([image_embeddings(imgs[j:j + 32]) for j in range(0, len(imgs), 32)]))
        emb.columns = [f"e{k}" for k in emb.columns]
        feito = pd.concat([feito, emb.assign(post_id=ids)], ignore_index=True)
        feito.to_parquet(CACHE, index=False)
        print(f"{min(i + 256, len(falta))}/{len(falta)} embeddings", flush=True)


if __name__ == "__main__":
    main()
