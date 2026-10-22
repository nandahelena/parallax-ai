"""F6.5: cliente MCP real (processo separado, stdio), como a persona da Galaxies consumiria o Parallax.

Usa langchain-mcp-adapters com UMA sessão (sem isso, cada chamada abre um processo novo; troubleshooting da liga).

    uv run python scripts/11_cliente_mcp.py
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from parallax import config


async def main() -> None:
    cliente = MultiServerMCPClient({"parallax": {
        "command": sys.executable, "args": [str(config.ROOT / "src" / "parallax" / "mcp_server.py")],
        "transport": "stdio"}})
    async with cliente.session("parallax") as sessao:
        tools = {t.name: t for t in await load_mcp_tools(sessao)}
        print("tools:", sorted(tools))
        agora = datetime(2026, 10, 24, 18, 0)
        req = {"creative": "Lançamento da nova paleta: antes e depois em 15s", "caption": "Nova paleta 💄\nlink na bio #shorts",
               "planned_datetime": agora.isoformat(), "sector": "beleza", "analisar_video": False,
               "channel_stats": {"subscribers": 250000, "video_count": 900, "is_brand": True, "language": "pt",
                                 "recent_posts": [{"published_at": (agora - timedelta(days=35 + 3 * i)).isoformat(),
                                                   "likes": 800 + 37 * i} for i in range(25)]}}
        r = json.loads((await tools["prever_criativo"].ainvoke({"request": req}))[0]["text"])
        print("previsão:", r["predicted_metric"], f"[{r['interval_low']}, {r['interval_high']}]", "|", r["explanation"])
        alt = {**req, "caption": "Antes e depois INCRÍVEL 😱\nComenta qual cor você usaria! #shorts"}
        rk = json.loads((await tools["ranquear_criativos"].ainvoke({"requests": [req, alt]}))[0]["text"])
        print("ranking:", [(i["posicao"], i["predicted_metric"]) for i in rk["ranking"]],
              "| P(1º > 2º) =", rk["prob_primeiro_supera_segundo"])


if __name__ == "__main__":
    asyncio.run(main())
