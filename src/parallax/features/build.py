"""Registro de features por bloco (PLANO regra 5) e montagem da tabela tabular.

Cada bloco declara de qual campo do contrato de entrada ele sai, ou seja, tudo existe antes da publicação.
"""

import numpy as np
import pandas as pd

from parallax import config
from parallax.data.history import channel_history
from parallax.features.text import caption_features

BLOCOS = {
    "canal": ("channel_id/channel_stats (posts anteriores + inscritos)", [
        "ch_n_prev", "ch_med_prev20", "ch_mean_prev5", "ch_mean_prev20", "ch_std_prev20", "ch_trend", "ch_last",
        "ch_days_since_last", "ch_posts_30d", "ch_log_subs", "ch_log_videos", "ch_is_brand", "ch_lang_br",
        "sector"]),
    "tempo": ("planned_datetime (+ idade de controle = H na inferência)", [
        "tm_hour", "tm_weekday", "tm_weekend", "tm_month", "tm_log_age_days"]),
    "metadados": ("creative (arquivo de mídia)", ["md_duration_s"]),
    "texto": ("caption", ["tx_title_len", "tx_title_words", "tx_desc_len", "tx_hashtags", "tx_has_shorts_tag",
                          "tx_mentions", "tx_links", "tx_emojis", "tx_question", "tx_exclaim", "tx_upper_ratio",
                          "tx_cta"]),
    # texto_emb: componentes de PCA ajustado no treino, criadas no treino (tx_pca_00..)
}


ROTULOS = {
    "ch_n_prev": "volume de posts anteriores do canal", "ch_med_prev20": "patamar recente de curtidas do canal",
    "ch_mean_prev5": "desempenho dos últimos 5 posts", "ch_mean_prev20": "média dos últimos 20 posts",
    "ch_std_prev20": "instabilidade do canal", "ch_trend": "tendência recente do canal",
    "ch_last": "desempenho do post anterior", "ch_days_since_last": "dias desde o último post",
    "ch_posts_30d": "frequência de postagem (30 dias)", "ch_log_subs": "inscritos do canal",
    "ch_log_videos": "tamanho do acervo do canal", "ch_is_brand": "canal de marca (vs criador)",
    "ch_lang_br": "canal brasileiro", "sector": "setor", "tm_hour": "horário de publicação",
    "tm_weekday": "dia da semana", "tm_weekend": "fim de semana", "tm_month": "mês",
    "tm_log_age_days": "horizonte de medição", "md_duration_s": "duração do vídeo",
    "tx_title_len": "tamanho do título", "tx_title_words": "palavras no título",
    "tx_desc_len": "tamanho da descrição", "tx_hashtags": "hashtags", "tx_has_shorts_tag": "#shorts na legenda",
    "tx_mentions": "menções (@)", "tx_links": "links na legenda", "tx_emojis": "emojis",
    "tx_question": "pergunta no título", "tx_exclaim": "exclamação no título",
    "tx_upper_ratio": "uso de caixa alta", "tx_cta": "chamada para ação",
    "tx_pca": "tema e conteúdo da legenda",
    "midia": "conteúdo do vídeo e do áudio (camada multimodal)",
}


def tabular_features(base: pd.DataFrame) -> pd.DataFrame:
    local = base["published_at"].dt.tz_convert(config.TZ_LOCAL)
    idade = (base["collected_at"] - base["published_at"]).dt.total_seconds() / 86400
    return pd.concat([
        channel_history(base),
        pd.DataFrame({
            "ch_log_subs": np.log1p(base["channel_stats_at_collection"].astype(float)),
            "ch_log_videos": np.log1p(base["channel_videos_at_collection"].astype(float)),
            "ch_is_brand": (base["channel_type"] == "marca").astype(float),
            "ch_lang_br": base["channel_lang"].str.startswith("BR").astype(float),
            "sector": base["sector"].astype(pd.CategoricalDtype(list(config.SETORES))),
            "tm_hour": local.dt.hour, "tm_weekday": local.dt.weekday,
            "tm_weekend": (local.dt.weekday >= 5).astype(float), "tm_month": local.dt.month,
            "tm_log_age_days": np.log(idade),
            "md_duration_s": base["duration_s"],
        }, index=base.index),
        caption_features(base["title"], base["description"]),
    ], axis=1)


def cols(*blocos: str) -> list[str]:
    return [c for b in blocos for c in BLOCOS[b][1]]
