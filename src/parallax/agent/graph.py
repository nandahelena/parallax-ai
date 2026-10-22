"""Agente orquestrador (TAPI §7): StateGraph determinístico sobre as 5 tools de `parallax.inference`.

T1 extrair atributos → T3 prever → T2 comparáveis (mais evidência se a confiança é baixa) → T4 explicar
(+ subagente LLM validado) → T5 ordenar (se houver 2+ criativos). Cada passo fica no estado, para inspeção.
O número sai sempre da T3 (LightGBM); o LLM só aparece na T4 e não pode alterar números.
"""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from parallax import inference as T
from parallax.schemas import PredictionRequest, PredictionResponse, RankResponse


class Estado(TypedDict, total=False):
    pedidos: list[PredictionRequest]
    narrar: bool
    atributos: list[dict[str, Any]]
    previsoes: list[dict[str, Any]]
    comparaveis: list[list]
    respostas: list[PredictionResponse]
    ranking: RankResponse | None
    trilha: list[str]


def _atributos(s: Estado) -> Estado:
    return {"atributos": [T.extrair_atributos(r) for r in s["pedidos"]], "trilha": [*s["trilha"], "T1 extrair_atributos"]}


def _prever(s: Estado) -> Estado:
    return {"previsoes": [T.prever(a["x"], a["x_midia"]) for a in s["atributos"]], "trilha": [*s["trilha"], "T3 prever (LightGBM)"]}


def _comparaveis(s: Estado) -> Estado:
    ks = [8 if p["confidence_flag"] else 5 for p in s["previsoes"]]  # baixa confiança: mostrar mais evidência
    comps = [T.comparaveis(r.sector, a["z"], k) for r, a, k in zip(s["pedidos"], s["atributos"], ks)]
    return {"comparaveis": comps, "trilha": [*s["trilha"], f"T2 comparaveis (k={ks})"]}


def _explicar(s: Estado) -> Estado:
    resp = [T.montar_resposta(a, p, T.explicar(a["x"], x_midia=a["x_midia"]), c, s.get("narrar", False))
            for a, p, c in zip(s["atributos"], s["previsoes"], s["comparaveis"])]
    return {"respostas": resp, "trilha": [*s["trilha"], "T4 explicar (SHAP" + (" + subagente LLM)" if s.get("narrar") else ")")]}


def _ordenar(s: Estado) -> Estado:
    return {"ranking": T.ordenar(s["respostas"], [r.creative for r in s["pedidos"]]),
            "trilha": [*s["trilha"], "T5 ordenar"]}


g = StateGraph(Estado)
for nome, fn in [("atributos", _atributos), ("prever", _prever), ("comparaveis", _comparaveis),
                 ("explicar", _explicar), ("ordenar", _ordenar)]:
    g.add_node(nome, fn)
g.add_edge(START, "atributos")
g.add_edge("atributos", "prever")
g.add_edge("prever", "comparaveis")
g.add_edge("comparaveis", "explicar")
g.add_conditional_edges("explicar", lambda s: "ordenar" if len(s["pedidos"]) > 1 else END, ["ordenar", END])
g.add_edge("ordenar", END)
agente = g.compile()


def executar(pedidos: list[PredictionRequest], narrar: bool = False) -> Estado:
    """Roda o agente para 1+ criativos. Devolve o estado completo (respostas, ranking e trilha de passos)."""
    return agente.invoke({"pedidos": pedidos, "narrar": narrar, "trilha": []})
