"""Servidor MCP do Parallax (FastMCP 3.4.7, stdio). As tools rodam o agente LangGraph (`parallax.agent.graph`).

    uv run python src/parallax/mcp_server.py
    npx @modelcontextprotocol/inspector uv run python src/parallax/mcp_server.py
"""

from fastmcp import FastMCP

from parallax import inference
from parallax.agent.graph import executar
from parallax.schemas import PredictionRequest, PredictionResponse, RankResponse, WhatIfItem

mcp = FastMCP(
    "parallax",
    instructions=(
        "Parallax prevê, ANTES da publicação, quantas curtidas um YouTube Short deve ter (moda, beleza, automotivo). "
        "Os números vêm de um modelo estatístico treinado em ~19 mil Shorts reais; não os altere nem invente outros. "
        "Use `relative_to_channel` para comparar com o desempenho típico do canal e `confidence_flag` para saber "
        "quando desconfiar. Para escolher entre criativos, prefira `ranquear_criativos`."
    ),
)


@mcp.tool
def prever_criativo(request: PredictionRequest, explicar_com_llm: bool = False) -> PredictionResponse:
    """Prevê as curtidas de um Short antes de publicar, com intervalo de 80%, fatores que pesaram, comparáveis
    reais e uma explicação. Informe o canal por `channel_id` (busca o histórico no YouTube) ou por `channel_stats`
    (inscritos + posts anteriores com curtidas), a legenda planejada, a data/hora planejada e o setor.
    `explicar_com_llm=True` gera a explicação em linguagem natural (validada contra os números do modelo)."""
    return executar([request], explicar_com_llm)["respostas"][0]


@mcp.tool
def ranquear_criativos(requests: list[PredictionRequest], explicar_com_llm: bool = False) -> RankResponse:
    """Compara 2+ criativos candidatos (ex.: variações de uma campanha) e os ordena do mais ao menos promissor,
    com a probabilidade de o 1º superar o 2º. Use antes de decidir qual publicar ou impulsionar."""
    return executar(requests, explicar_com_llm)["ranking"]


@mcp.tool
def simular_what_if(request: PredictionRequest, variacoes: list[dict[str, str]]) -> list[WhatIfItem]:
    """Testa alternativas antes de publicar: cada variação troca `caption` e/ou `planned_datetime` (ISO 8601)
    e devolve as curtidas previstas e a diferença % em relação ao original."""
    return inference.simular_what_if(request, variacoes)


@mcp.resource("parallax://modelo")
def modelo() -> dict:
    """Versão, período de treino e métricas de teste do modelo em uso."""
    m = inference.artefatos()["meta"]
    return {k: m[k] for k in ("versao", "base", "cortes", "metricas_teste", "cobertura_teste_bruta", "horizonte_dias")}


if __name__ == "__main__":
    mcp.run()
