"""Subagente explicador (TAPI §3): transforma o JSON do modelo em linguagem natural, no tom "acho que vai performar
assim por conta disso, disso e disso". Nunca produz números: o validador reprova qualquer número ou fator que não
esteja no JSON, com uma segunda chance e depois fallback para o template."""

from pydantic import BaseModel

from parallax import config
from parallax.explain.validator import validar
from parallax.schemas import PredictionResponse

PROMPT = """Você é o subagente explicador do Parallax, um modelo estatístico que prevê curtidas de YouTube Shorts antes
da publicação. Abaixo está o JSON da previsão: todos os números vieram do modelo.

Escreva de 2 a 3 frases em português, no tom: "Acho que este criativo vai performar <acima/abaixo/em linha com> o
padrão do canal, com cerca de N curtidas, por conta de A, B e C."
Regras obrigatórias:
- Use SOMENTE números que aparecem no JSON, escritos como estão (ex.: 1066 curtidas, intervalo de 559 a 2907, 0,93×).
- Cite SOMENTE fatores de `top_drivers`, com o nome exato, respeitando o sentido (aumenta/reduz).
- Não invente porcentagens, médias ou fatores. Não arredonde para "mil".
- Se `confidence_flag` for true, diga que a confiança é baixa e por quê.
Em `fatores_citados`, liste exatamente os nomes de fatores que você usou.

JSON:
{payload}
"""


class Narrativa(BaseModel):
    texto: str
    fatores_citados: list[str]


def _gerar(prompt: str, cfg, esperas=(5, 15)) -> Narrativa:
    """Chamada ao LLM com retry em indisponibilidade (5xx) e troca de modelo se um estiver fora ou sem cota (429)."""
    import time

    from google.genai import errors

    from parallax.features.vlm import _client
    ultimo = None
    for modelo in (config.GEMINI_LLM, config.GEMINI_VLM):
        for espera in (*esperas, None):
            try:
                r = _client().models.generate_content(model=modelo, contents=prompt, config=cfg)
                return r.parsed or Narrativa.model_validate_json(r.text)
            except errors.ServerError as e:
                ultimo = e
                if espera is None:
                    break
                time.sleep(espera)
            except errors.ClientError as e:
                if e.code != 429:
                    raise
                ultimo = e
                break  # cota do dia esgotada neste modelo: tenta o próximo
    raise ultimo


def narrar(resp: PredictionResponse, tentativas: int = 2) -> tuple[str | None, str]:
    """(explicação validada, status). Texto None → o chamador usa o template; o status diz o porquê."""
    try:
        from google.genai import types
        payload = resp.model_dump_json(include={"predicted_metric", "interval_low", "interval_high",
                                                "relative_to_channel", "confidence_flag", "confidence_reasons",
                                                "top_drivers"}, indent=1)
        prompt = PROMPT.format(payload=payload)
        cfg = types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json",
                                          response_schema=Narrativa)
        problemas = []
        for _ in range(tentativas):
            n = _gerar(prompt, cfg)
            problemas = validar(n.texto, n.fatores_citados, resp)
            if not problemas:
                return n.texto, "explicação gerada pelo LLM e aprovada no validador"
            prompt += f"\n\nSua resposta anterior foi reprovada: {problemas}. Corrija seguindo as regras."
        return None, f"explicação do LLM reprovada no validador ({problemas[:2]}); usando o template"
    except Exception as e:  # noqa: BLE001 - explicação é acessória: qualquer falha cai no template, nunca quebra a tool
        return None, f"LLM indisponível ({type(e).__name__}); usando o template"
