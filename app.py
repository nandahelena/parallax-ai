"""Interface do Parallax (opcional no TAPI; a entrega obrigatória é a tool/MCP).

Só embrulha `parallax.inference`: nenhum número é calculado aqui.

    uv run streamlit run app.py
"""

from datetime import date, datetime, time

import pandas as pd
import streamlit as st

from parallax import config
from parallax.schemas import ChannelStats, PostAnterior, PredictionRequest

st.set_page_config(page_title="Parallax", page_icon="docs/assets/logo.svg", layout="wide")


@st.cache_resource(show_spinner="Carregando o modelo...")
def _tool():
    from parallax.agent.graph import executar
    from parallax.inference import artefatos
    artefatos()  # aquece os artefatos para a primeira previsão não demorar
    return executar


@st.cache_data(show_spinner=False)
def _canais_demo() -> pd.DataFrame:
    """Canais que o modelo nunca viu, para demonstrar com histórico e métrica real conhecidos."""
    b = pd.read_parquet(config.PROCESSED / config.BASE_TREINO)
    return b[b["role"] == "demo"]


def _stats(g: pd.DataFrame, antes: pd.Timestamp) -> ChannelStats:
    h = g[g["published_at"] < antes].sort_values("published_at").tail(60)
    r = g.iloc[0]
    return ChannelStats(
        subscribers=int(r["channel_stats_at_collection"]) if pd.notna(r["channel_stats_at_collection"]) else None,
        video_count=int(r["channel_videos_at_collection"]) if pd.notna(r["channel_videos_at_collection"]) else None,
        is_brand=r["channel_type"] == "marca", language="pt" if str(r["channel_lang"]).startswith("BR") else "en",
        recent_posts=[PostAnterior(published_at=t, likes=int(l)) for t, l in zip(h["published_at"], h["metric_target"])])


st.logo("docs/assets/logo.svg", size="large")
st.title("Parallax")
st.caption("Quantas curtidas este Short vai ter? Pergunte **antes** de publicar. "
           "O número vem do modelo treinado; o LLM só explica.")

demo = _canais_demo()
nomes = pd.read_csv(config.CANAIS_CSV).dropna(subset=["channel_id"]).set_index("channel_id")["titulo_api"].to_dict()

with st.sidebar:
    st.subheader("Canal")
    modo = st.radio("Origem do histórico", ["Canal de demonstração", "Informar manualmente"],
                    help="Os canais de demonstração nunca foram vistos no treino e têm métrica real conhecida.")
    if modo == "Canal de demonstração":
        cid = st.selectbox("Canal", sorted(demo["channel_id"].unique()),
                           format_func=lambda c: f"{nomes.get(c, c)} · {demo[demo.channel_id == c].sector.iloc[0]}")
        g = demo[demo["channel_id"] == cid]
        setor = g["sector"].iloc[0]
        st.metric("Inscritos", f"{int(g['channel_stats_at_collection'].iloc[0]):,}".replace(",", "."))
        st.metric("Shorts na base", len(g))
    else:
        setor = st.selectbox("Setor", config.SETORES)
        inscritos = st.number_input("Inscritos", 1000, 100_000_000, 250_000, step=1000)
        marca = st.toggle("É canal de marca", True)
        curtidas = st.text_area("Curtidas dos últimos Shorts (uma por linha, do mais antigo ao mais novo)",
                                "800\n1200\n640\n1500\n980\n1100\n720\n1350", height=160)
    st.divider()
    narrar = st.toggle("Explicação por LLM", False, help="Gemini, validado contra o JSON do modelo. Gasta cota.")
    video = st.toggle("Analisar o vídeo", False, help="Baixa e processa o Short (~20 s). Os atributos são expostos "
                                                      "para inspeção; o número não muda (ver ablação).")

st.subheader("Criativos candidatos")
st.caption("Preencha 1 para prever, ou 2+ para o Parallax ordenar do mais ao menos promissor.")

if modo == "Canal de demonstração":
    recentes = g.sort_values("published_at").tail(4)
    padrao = pd.DataFrame({"legenda": recentes["title"], "criativo": recentes["media_ref"],
                           "curtidas reais": recentes["metric_target"]}).reset_index(drop=True)
else:
    padrao = pd.DataFrame({"legenda": ["Antes e depois em 15s 💄\nlink na bio #shorts", ""],
                           "criativo": ["", ""], "curtidas reais": [None, None]})

tabela = st.data_editor(padrao, num_rows="dynamic", width="stretch", column_config={
    "legenda": st.column_config.TextColumn("Legenda planejada", width="large", required=True),
    "criativo": st.column_config.TextColumn("URL do Short ou descrição", width="medium"),
    "curtidas reais": st.column_config.NumberColumn("Curtidas reais (só para conferir)", width="small")})

c1, c2 = st.columns(2)
dia = c1.date_input("Data planejada", date(2026, 10, 24))
hora = c2.time_input("Hora planejada", time(18, 0))
quando = datetime.combine(dia, hora)

if st.button("Prever", type="primary", width="stretch"):
    linhas = [r for r in tabela.to_dict("records") if str(r.get("legenda", "")).strip()]
    if not linhas:
        st.warning("Preencha ao menos uma legenda.")
        st.stop()
    stats = _stats(g, pd.Timestamp(quando, tz="UTC")) if modo == "Canal de demonstração" else ChannelStats(
        subscribers=int(inscritos), video_count=None, is_brand=marca, language="pt",
        recent_posts=[PostAnterior(published_at=pd.Timestamp(quando) - pd.Timedelta(days=35 + 4 * i), likes=int(v))
                      for i, v in enumerate(reversed([x for x in curtidas.split() if x.strip().isdigit()]))])
    try:
        pedidos = [PredictionRequest(creative=(r.get("criativo") or r["legenda"]), caption=r["legenda"],
                                     planned_datetime=quando, sector=setor, channel_stats=stats,
                                     analisar_video=video) for r in linhas]
    except Exception as e:  # noqa: BLE001 - erro de contrato vira mensagem, não traceback
        st.error(f"Entrada inválida: {e}")
        st.stop()

    with st.spinner("Extraindo atributos, consultando o modelo e montando a evidência..."):
        estado = _tool()(pedidos, narrar)
    respostas, ranking = estado["respostas"], estado.get("ranking")
    st.caption("Trilha do agente: " + " → ".join(estado["trilha"]))

    if ranking:
        st.subheader("Ranking")
        rk = pd.DataFrame([{"#": i.posicao, "criativo": linhas[i.indice]["legenda"].split("\n")[0][:60],
                            "previsto": round(i.predicted_metric), "intervalo de 80%": f"{i.interval_low:,.0f} – {i.interval_high:,.0f}".replace(",", "."),
                            "vs. canal": f"{i.relative_to_channel}×",
                            "curtidas reais": linhas[i.indice].get("curtidas reais")} for i in ranking.ranking])
        st.dataframe(rk, hide_index=True, width="stretch")
        if ranking.prob_primeiro_supera_segundo is not None:
            st.info(f"Probabilidade de o 1º colocado superar o 2º: **{ranking.prob_primeiro_supera_segundo:.0%}**")

    for r, linha in zip(respostas, linhas):
        with st.expander(linha["legenda"].split("\n")[0][:80] or "criativo", expanded=len(respostas) == 1):
            m1, m2, m3 = st.columns(3)
            m1.metric("Curtidas previstas", f"{r.predicted_metric:,.0f}".replace(",", "."),
                      help=f"em {config.HORIZONTE_DIAS} dias")
            m2.metric("Intervalo de 80%", f"{r.interval_low:,.0f} – {r.interval_high:,.0f}".replace(",", "."))
            m3.metric("Em relação ao canal", f"{r.relative_to_channel}×")
            if linha.get("curtidas reais") and pd.notna(linha["curtidas reais"]):
                real = int(linha["curtidas reais"])
                dentro = r.interval_low <= real <= r.interval_high
                st.success(f"Curtidas reais: **{real:,}**".replace(",", ".") +
                           (" — dentro do intervalo previsto ✓" if dentro else " — fora do intervalo"))
            if r.confidence_flag:
                st.warning("Baixa confiança: " + "; ".join(r.confidence_reasons))
            st.write(r.explanation)

            a, b = st.columns(2)
            a.markdown("**Fatores que mais pesaram**")
            a.dataframe(pd.DataFrame([{"fator": d.factor, "efeito": d.direction, "peso": d.weight}
                                      for d in r.top_drivers]), hide_index=True, width="stretch")
            b.markdown("**Criativos comparáveis (reais)**")
            b.dataframe(pd.DataFrame([{"Short": c.title[:40], "curtidas": c.likes, "vs. canal": f"{c.relative_to_channel}×",
                                       "link": c.url} for c in r.comparables]), hide_index=True, width="stretch",
                        column_config={"link": st.column_config.LinkColumn("link", display_text="abrir")})

            midia = {k.removeprefix("midia_"): v for k, v in r.features_used.items() if k.startswith("midia_")}
            if midia:
                st.markdown("**Atributos extraídos do vídeo** (expostos para inspeção; não entram no número)")
                st.json(midia, expanded=False)
            with st.popover("Ver todas as features e avisos"):
                st.json({"model_version": r.model_version, "warnings": r.warnings,
                         "features_used": r.features_used}, expanded=False)
