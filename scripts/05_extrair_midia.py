"""F3.1–F3.2: camada multimodal 1 sobre o subconjunto. Retomável: um vídeo com media.json nunca é reprocessado.

Por vídeo: baixa vídeo (360p, H.264) e áudio separados → extrai features → grava data/media/<id>/ → apaga a mídia.
O download do próximo vídeo (com pausa de 5-10 s, guia da liga) roda enquanto o atual é processado.

    uv run python scripts/05_extrair_midia.py [--limite N]
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yt_dlp

from parallax import config
from parallax.features.media import extrair
from parallax.rede import erro_de_rede

SUBSET = config.PROCESSED / "subset_multimodal.parquet"
FALHAS = config.MEDIA / "_falhas.jsonl"
YDL = {"format": "bv*,ba", "format_sort": ["res:360", "vcodec:h264", "abr"], "quiet": True, "no_warnings": True,
       "noprogress": True, "outtmpl": "%(id)s.%(acodec.0:4)s.%(ext)s"}


def subconjunto() -> pd.DataFrame:
    """Fixado em disco na primeira execução: demo → (teste completo + amostras de val e treino), embaralhado."""
    if SUBSET.exists():
        return pd.read_parquet(SUBSET)
    b = pd.read_parquet(config.PROCESSED / "base_v1.parquet")
    cortes = json.loads((config.MODELS / config.VERSAO_MODELO / "meta.json").read_text(encoding="utf-8"))["cortes"]
    ini_val, ini_teste = (pd.Timestamp(cortes[s][0], tz="UTC") for s in ("val", "test"))
    demo = b[b["role"] == "demo"].sort_values("published_at").groupby("channel_id").tail(10).assign(grupo="demo")
    t = b[b["role"] == "treino"]
    teste = t[t["published_at"] >= ini_teste].assign(grupo="teste")
    val = t[(t["published_at"] >= ini_val) & (t["published_at"] < ini_teste)].sample(600, random_state=config.SEED)
    treino = t[t["published_at"] < ini_val].sample(1500, random_state=config.SEED)
    resto = pd.concat([teste, val.assign(grupo="val"), treino.assign(grupo="treino")]).sample(frac=1, random_state=config.SEED)
    s = pd.concat([demo, resto])[["post_id", "grupo"]].reset_index(drop=True)
    s.to_parquet(SUBSET, index=False)
    return s


def baixar(vid: str) -> Path:
    time.sleep(random.uniform(5, 10))
    pasta = config.MEDIA / vid
    pasta.mkdir(parents=True, exist_ok=True)
    with yt_dlp.YoutubeDL({**YDL, "paths": {"home": str(pasta)}}) as y:
        y.download([f"https://www.youtube.com/shorts/{vid}"])
    return pasta


def processar(vid: str, pasta: Path) -> None:
    video = next(pasta.glob(f"{vid}.none.*"), None)
    som = next(pasta.glob(f"{vid}.mp4a.*"), None) or next(pasta.glob(f"{vid}.opus.*"), None)
    if video is None:
        raise FileNotFoundError("stream de vídeo não baixado")
    try:
        f, emb, txt = extrair(video, som)
        np.save(pasta / "frames_emb.npy", emb.astype(np.float32))
        (pasta / "media.json").write_text(json.dumps({**f, "transcricao": txt}, ensure_ascii=False), encoding="utf-8")
    finally:
        for m in pasta.glob(f"{vid}.*"):  # orçamento de disco: só as features ficam (PLANO regra 12)
            m.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--parte", default="0/1", help="k/n: processa só a k-ésima de n partes (processos em paralelo)")
    ap.add_argument("--threads", type=int, default=6, help="threads de torch/OpenCV por processo")
    args = ap.parse_args()
    k, n = map(int, args.parte.split("/"))
    import cv2
    import torch
    torch.set_num_threads(args.threads)  # sem limite, cada processo tenta usar todos os núcleos e todos ficam lentos
    cv2.setNumThreads(2)
    s = subconjunto()
    falhou = set()
    if FALHAS.exists():
        falhou = {json.loads(l)["post_id"] for l in FALHAS.read_text(encoding="utf-8").splitlines()}
    fila = [(r.post_id, r.grupo) for r in s.itertuples()
            if not (config.MEDIA / r.post_id / "media.json").exists() and r.post_id not in falhou
            and r.Index % n == k and r.Index < config.N_SUBSET_MULTIMODAL][: args.limite]
    print(f"subconjunto {len(s)} | pendentes {len(fila)}", flush=True)

    seguidas, t0, espera, extracao = 0, time.time(), 0.0, 0.0
    with ThreadPoolExecutor(1) as pool:
        prox = pool.submit(baixar, fila[0][0]) if fila else None
        for i, (vid, grupo) in enumerate(fila):
            atual = prox
            prox = pool.submit(baixar, fila[i + 1][0]) if i + 1 < len(fila) else None
            try:
                ta = time.time()
                pasta = atual.result()
                tb = time.time()
                processar(vid, pasta)
                espera += tb - ta
                extracao += time.time() - tb
                seguidas = 0
            except Exception as e:  # noqa: BLE001 - um vídeo ruim não pode derrubar a extração inteira
                seguidas += 1
                for m in (config.MEDIA / vid).glob(f"{vid}.*"):  # download parcial não fica no disco
                    m.unlink()
                if erro_de_rede(e):  # sem internet: não registra falha, o vídeo volta na próxima execução
                    if seguidas >= 10:
                        print("10 erros de rede seguidos: parando. Rode de novo quando a conexão voltar.", flush=True)
                        break
                    time.sleep(60)
                    continue
                with FALHAS.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"post_id": vid, "erro": str(e)[:300]}, ensure_ascii=False) + "\n")
                if seguidas >= 24:
                    print("bloqueio persistente; parando. Apague _falhas.jsonl e rode de novo mais tarde.", flush=True)
                    break
                if seguidas % 8 == 0:  # provável bloqueio do YouTube ("not a bot"): uma pausa longa a cada 8
                    print(f"{seguidas} falhas seguidas: pausando 15 min", flush=True)
                    time.sleep(900)
            if (i + 1) % 5 == 0:
                print(f"{i + 1}/{len(fila)} | {(time.time() - t0) / (i + 1):.1f} s/vídeo "
                      f"(espera download {espera / (i + 1):.1f} s, extração {extracao / (i + 1):.1f} s)", flush=True)


if __name__ == "__main__":
    main()
