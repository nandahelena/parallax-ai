"""F5a.4: contrato, low ≤ pred ≤ high e "o número da tool é o número do modelo". Sem rede (usa channel_stats)."""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pytest

from parallax import config
from parallax.schemas import ChannelStats, PostAnterior, PredictionRequest

pytestmark = pytest.mark.skipif(not (config.MODELS / config.VERSAO_MODELO / "meta.json").exists(),
                                reason="sem artefatos: rode scripts/07_treinar.py")


def _req(**kw):
    agora = datetime(2026, 10, 20, 18, 0)
    posts = [PostAnterior(published_at=agora - timedelta(days=35 + 3 * i), likes=800 + 37 * i) for i in range(25)]
    return PredictionRequest(**{
        "creative": "Tutorial de make de 30s com antes e depois", "caption": "Make em 30s 💄\nlink na bio #shorts",
        "planned_datetime": agora, "sector": "beleza",
        "channel_stats": ChannelStats(subscribers=250_000, video_count=900, recent_posts=posts), **kw})


def test_contrato_exige_canal():
    with pytest.raises(ValueError):
        PredictionRequest(creative="x", caption="y", planned_datetime=datetime.now(), sector="moda")


def test_intervalo_ordenado_e_numero_vem_do_modelo():
    from parallax.inference import artefatos, montar_features, prever_criativo

    req = _req()
    r = prever_criativo(req)
    assert r.interval_low <= r.predicted_metric <= r.interval_high
    x, _, _ = montar_features(req)
    esperado = np.expm1(x["ch_med_prev20"].iloc[0] + artefatos()["ponto"].predict(x)[0])
    assert np.isclose(r.predicted_metric, esperado, rtol=1e-3)
    assert len(r.top_drivers) == 5 and len(r.comparables) == 5


def test_mcp_lista_e_chama_a_tool():
    from fastmcp import Client

    from parallax.inference import prever_criativo
    from parallax.mcp_server import mcp

    async def run():
        async with Client(mcp) as c:
            assert "prever_criativo" in [t.name for t in await c.list_tools()]
            res = await c.call_tool("prever_criativo", {"request": _req().model_dump(mode="json")})
            return res.structured_content

    out = asyncio.run(run())
    assert np.isclose(out["predicted_metric"], prever_criativo(_req()).predicted_metric)
