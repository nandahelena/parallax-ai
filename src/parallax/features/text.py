"""Bloco texto: medidas da legenda + embedding multilíngue (cacheado por post_id)."""

from functools import cache

import numpy as np
import pandas as pd

from parallax import config

CTA = (r"(?i)(link na bio|link in bio|comente|comenta|segue|siga|inscreva|compre|confira|shop now|subscribe|"
       r"follow|comment|buy now|check out|tag)")
EMOJI = "[\U0001F300-\U0001FAFF☀-➿]"  # sem r: caracteres reais, pois o RE2 (pandas 3 + Arrow) não aceita \U


def caption_features(title: pd.Series, description: pd.Series) -> pd.DataFrame:
    full = title + "\n" + description
    letras = full.str.count(r"[A-Za-zÀ-ÿ]").clip(lower=1)
    return pd.DataFrame({
        "tx_title_len": title.str.len(),
        "tx_title_words": title.str.split().str.len(),
        "tx_desc_len": description.str.len(),
        "tx_hashtags": full.str.count(r"#\w+"),
        "tx_has_shorts_tag": full.str.contains("#shorts", case=False),
        "tx_mentions": full.str.count(r"@\w+"),
        "tx_links": full.str.count(r"https?://"),
        "tx_emojis": full.str.count(EMOJI),
        "tx_question": title.str.count(r"\?"),
        "tx_exclaim": title.str.count("!"),
        "tx_upper_ratio": full.str.count(r"[A-ZÀ-Þ]") / letras,
        "tx_cta": full.str.count(CTA),
    }).astype(float)


def text_embeddings(post_id: pd.Series, title: pd.Series, description: pd.Series) -> np.ndarray:
    """Embeddings de título + começo da descrição; calcula só o que falta no cache."""
    arq = config.PROCESSED / "emb_texto.parquet"
    old = pd.read_parquet(arq) if arq.exists() else pd.DataFrame(columns=["post_id"])
    falta = ~post_id.isin(old["post_id"])
    if falta.any():
        emb = encode(title[falta], description[falta])
        new = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])]).assign(
            post_id=post_id[falta].to_numpy())
        old = pd.concat([old, new], ignore_index=True)
        old.to_parquet(arq, index=False)
    return old.set_index("post_id").loc[post_id].to_numpy(dtype=np.float32)


@cache
def _model():
    from sentence_transformers import SentenceTransformer  # import pesado só quando precisa

    return SentenceTransformer(config.TEXT_EMB_MODEL, device="cpu")


def encode(title: pd.Series, description: pd.Series) -> np.ndarray:
    """Mesmo texto e mesmo modelo no treino e na inferência (paridade, PLANO regra 6)."""
    txt = (title + "\n" + description.str[:300]).tolist()
    return _model().encode(txt, batch_size=64, show_progress_bar=len(txt) > 256, normalize_embeddings=True)
