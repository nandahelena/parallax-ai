"""Camada multimodal 2: VLM responde a rubrica fixa em JSON (TAPI §6.2). Gemini lê a URL pública do YouTube direto."""

import random
import time
from functools import cache

from parallax import config
from parallax.schemas import Rubrica

PROMPT = (
    "Você é um analista de criativos de vídeo curto. Assista ao YouTube Short inteiro (imagem e som) e preencha a "
    "rubrica de forma objetiva, sem adivinhar o desempenho real do vídeo. Notas de 0 a 10 comparam com Shorts de "
    "marcas em geral. Campos de setor que não se aplicam ao vídeo devem ser false. Responda em português."
)


class CotaEsgotada(RuntimeError):
    """Limite diário do free tier atingido: retomar amanhã."""


@cache
def _client():
    from google import genai
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY vazia no .env")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def rubrica(url: str, modelo: str = config.GEMINI_VLM, tentativas: int = 5) -> Rubrica:
    """Uma chamada (com retry) ao VLM. Levanta CotaEsgotada no limite diário e ValueError se o JSON vier inválido."""
    from google.genai import errors, types

    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_schema=Rubrica,
                                      media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW)
    conteudo = types.Content(parts=[types.Part(file_data=types.FileData(file_uri=url)), types.Part(text=PROMPT)])
    for k in range(tentativas):
        try:
            resp = _client().models.generate_content(model=modelo, contents=conteudo, config=cfg)
            if resp.parsed is None:  # google-genai 2.25 falha em silêncio se o JSON viola o schema
                return Rubrica.model_validate_json(resp.text)
            return resp.parsed
        except errors.ClientError as e:
            if e.code != 429:
                raise
            if "PerDay" in str(e) or "per day" in str(e).lower():
                raise CotaEsgotada(str(e)[:200]) from e
            time.sleep(min(60, 2 ** k * 5) + random.uniform(0, 3))  # limite por minuto: backoff exponencial
        except errors.ServerError:
            time.sleep(2 ** k * 5)
    raise RuntimeError(f"VLM sem resposta após {tentativas} tentativas")


TONS = ("ausente", "calmo", "animado", "instrucional")


def carregar(post_ids):
    """Rubrica → colunas vl_* numéricas + texto dos descritores (para clusterizar). vl_falhou=1 quando o VLM falhou."""
    import json

    import pandas as pd
    linhas = []
    for v in post_ids:
        p = config.MEDIA / v / "vlm.json"
        d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        if d is None:  # ainda sem rubrica (cota do VLM): desconhecido, não "ok"
            linhas.append({"vl_falhou": float("nan"), "vl_texto": ""})
            continue
        if d.get("vlm_falhou"):
            linhas.append({"vl_falhou": 1.0, "vl_texto": ""})
            continue
        r = {f"vl_{k}": float(x) for k, x in d.items() if isinstance(x, (bool, int, float))}
        r["vl_modelo_fallback"] = float(d.get("modelo", config.GEMINI_VLM) != config.GEMINI_VLM)  # D22
        r |= {f"vl_tom_{t}": float(d["tom_narracao"] == t) for t in TONS}
        r["vl_falhou"] = 0.0
        r["vl_texto"] = " | ".join([d["hook_primeiros_3s"], *d["elementos_video_top5"], *d["elementos_audio_top5"]])
        linhas.append(r)
    return pd.DataFrame(linhas, index=post_ids.index)
