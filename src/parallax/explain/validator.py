"""Validador determinístico da explicação (autoavaliação da liga): o texto só pode citar números e fatores que estão
no JSON do modelo. Sem LLM aqui: é a garantia de que o LLM não inventou nada."""

import re

from parallax.schemas import PredictionResponse

NUMERO = re.compile(r"\d[\d.,]*\d|\d")
CONSTANTES = (60, 80, 20)  # horizonte em dias, intervalo de 80%, mediana dos 20 posts anteriores


def permitidos(resp: PredictionResponse) -> list[float]:
    nums = [resp.predicted_metric, resp.interval_low, resp.interval_high, resp.relative_to_channel, *CONSTANTES]
    nums += [abs(d.weight) for d in resp.top_drivers]
    nums += [v for c in resp.comparables for v in (c.likes, c.relative_to_channel)]
    return nums


def _leituras(tok: str) -> set[float]:
    """Um número em texto pode estar em formato BR (1.066,5) ou US (1,066.5)."""
    out = set()
    for s in (tok.replace(".", "").replace(",", "."), tok.replace(",", ""), tok.replace(",", ".")):
        try:
            out.add(float(s))
        except ValueError:
            pass
    return out


def validar(texto: str, fatores_citados: list[str], resp: PredictionResponse) -> list[str]:
    """Lista de problemas (vazia = aprovado)."""
    ok = permitidos(resp)
    problemas = [f"número fora do JSON: {tok}" for tok in NUMERO.findall(texto)
                 if not any(abs(v - a) <= max(0.5, 0.01 * abs(a)) for v in _leituras(tok) for a in ok)]
    validos = {d.factor for d in resp.top_drivers}
    problemas += [f"fator fora dos top_drivers: {f}" for f in fatores_citados if f not in validos]
    return problemas
