"""F3.4: rubrica do VLM sobre o subconjunto multimodal (mesma ordem: demo primeiro). Retomável e limitado pela
cota diária do free tier: rode uma vez por dia até cobrir o subconjunto.

    uv run python scripts/06_extrair_vlm.py
"""

import json
import time

import pandas as pd

from parallax import config
from parallax.features.vlm import CotaEsgotada, rubrica
from parallax.rede import erro_de_rede


def main() -> None:
    s = pd.read_parquet(config.PROCESSED / "subset_multimodal.parquet")
    s = s.head(config.N_SUBSET_MULTIMODAL)  # D23: mesmos vídeos da camada 1
    fila = [v for v in s["post_id"] if not (config.MEDIA / v / "vlm.json").exists()]
    print(f"subconjunto {len(s)} | pendentes {len(fila)}", flush=True)
    feitos, sem_rede, t0 = 0, 0, time.time()
    modelos = [config.GEMINI_VLM, config.GEMINI_VLM_FALLBACK]  # D22: cotas diárias separadas por modelo
    for vid in fila:
        pasta = config.MEDIA / vid
        pasta.mkdir(parents=True, exist_ok=True)
        try:
            out = rubrica(f"https://www.youtube.com/shorts/{vid}", modelos[0]).model_dump() | {"modelo": modelos[0]}
            sem_rede = 0
        except CotaEsgotada as e:
            print(f"cota diária de {modelos[0]} esgotada após {feitos} vídeos: {str(e)[:80]}", flush=True)
            modelos.pop(0)
            if not modelos:
                break
            continue  # o vídeo volta na próxima execução (já com o modelo seguinte nesta)
        except Exception as e:  # noqa: BLE001 - extração falha vira flag (feature), não derruba o lote
            if "NOT_FOUND" in str(e):  # modelo indisponível é erro de configuração, não do vídeo
                print(f"{modelos[0]} indisponível: {str(e)[:80]}", flush=True)
                modelos.pop(0)
                if not modelos:
                    break
                continue
            if erro_de_rede(e):  # sem internet: não grava nada, o vídeo volta na próxima execução
                sem_rede += 1
                if sem_rede >= 10:
                    print("10 erros de rede seguidos: parando. Rode de novo quando a conexão voltar.", flush=True)
                    break
                time.sleep(60)
                continue
            out = {"vlm_falhou": True, "erro": str(e)[:300]}
        (pasta / "vlm.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        feitos += 1
        if feitos % 25 == 0:
            print(f"{feitos} feitos | {(time.time() - t0) / feitos:.1f} s/vídeo", flush=True)


if __name__ == "__main__":
    main()
