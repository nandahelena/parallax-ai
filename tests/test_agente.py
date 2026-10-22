"""G3: agente LangGraph, ranking, what-if e validador da explicação. Sem rede (channel_stats, sem LLM)."""

import asyncio

import numpy as np
import pytest

from parallax import config
from parallax.explain.validator import validar

pytestmark = pytest.mark.skipif(not (config.MODELS / config.VERSAO_MODELO / "meta.json").exists(),
                                reason="sem artefatos: rode scripts/07_treinar.py")


def _req(**kw):
    from test_inference import _req as base
    return base(**kw)


def test_agente_devolve_o_mesmo_numero_da_funcao_e_registra_os_5_passos():
    from parallax.agent.graph import executar
    from parallax.inference import prever_criativo

    a, b = _req(), _req(caption="Antes e depois incrível!\nSó assiste #shorts")
    s = executar([a, b])
    assert [r.predicted_metric for r in s["respostas"]] == [prever_criativo(a).predicted_metric,
                                                            prever_criativo(b).predicted_metric]
    assert [p.split()[0] for p in s["trilha"]] == ["T1", "T3", "T2", "T4", "T5"]
    rk = s["ranking"]
    assert [i.posicao for i in rk.ranking] == [1, 2]
    assert rk.ranking[0].predicted_metric >= rk.ranking[1].predicted_metric
    assert 0.5 <= rk.prob_primeiro_supera_segundo <= 1


def test_comparaveis_diversificados():
    from parallax.inference import prever_criativo
    r = prever_criativo(_req())
    assert len(r.comparables) == 5


def test_what_if_mostra_delta():
    from parallax.inference import simular_what_if
    out = simular_what_if(_req(), [{"planned_datetime": "2026-10-24T11:00:00"}])
    assert len(out) == 1 and np.isfinite(out[0].delta_vs_original_pct)


def test_validador_aprova_template_e_reprova_invencao():
    from parallax.inference import prever_criativo
    r = prever_criativo(_req())
    assert validar(r.explanation, [d.factor for d in r.top_drivers[:3]], r) == []
    ruim = validar(f"Vai bombar com 987654 curtidas, 300% acima do canal.", ["dancinha viral"], r)
    assert any("987654" in p for p in ruim) and any("300" in p for p in ruim)
    assert any("dancinha viral" in p for p in ruim)


def test_mcp_tem_as_tools_do_agente():
    from fastmcp import Client

    from parallax.mcp_server import mcp

    async def run():
        async with Client(mcp) as c:
            nomes = {t.name for t in await c.list_tools()}
            rk = await c.call_tool("ranquear_criativos",
                                   {"requests": [_req().model_dump(mode="json"), _req().model_dump(mode="json")]})
            return nomes, rk.structured_content

    nomes, rk = asyncio.run(run())
    assert {"prever_criativo", "ranquear_criativos", "simular_what_if"} <= nomes
    assert len(rk["ranking"]) == 2
