"""Contrato da tool (TAPI §7), validado com Pydantic. É agnóstico à base: `channel_stats` dispensa a nossa coleta."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Setor = Literal["moda", "beleza", "automotivo"]


class PostAnterior(BaseModel):
    published_at: datetime
    likes: int | None = Field(None, ge=0, description="curtidas; None se o post tem menos de 30 dias ou likes ocultos")


class ChannelStats(BaseModel):
    subscribers: int | None = Field(None, ge=0)
    video_count: int | None = Field(None, ge=0)
    is_brand: bool = True
    language: str = Field("pt", description="idioma principal do canal (pt, en, ...)")
    recent_posts: list[PostAnterior] = Field(..., min_length=3, description="Shorts anteriores (ideal: os 20+ mais recentes)")


class PredictionRequest(BaseModel):
    creative: str = Field(..., description="URL do YouTube, caminho de vídeo/imagem ou texto descrevendo o criativo")
    caption: str = Field(..., description="legenda planejada; a 1ª linha é tratada como título")
    planned_datetime: datetime
    sector: Setor
    channel_id: str | None = Field(None, description="ID do canal no YouTube (UC...); o histórico é buscado na API")
    channel_stats: ChannelStats | None = None
    analisar_video: bool = Field(True, description="baixar e analisar o vídeo (URL do YouTube ou arquivo); False = só canal, tempo e texto, mais rápido")

    @model_validator(mode="after")
    def _canal(self):
        if not (self.channel_id or self.channel_stats):
            raise ValueError("informe channel_id ou channel_stats")
        return self


Nota = Field(..., ge=0, le=10)


class Rubrica(BaseModel):
    """Questionário fixo respondido pelo VLM (TAPI §6.2). Campos `_0a10` são notas de 0 a 10."""
    hook_primeiros_3s: str = Field(..., description="o que acontece nos 3 primeiros segundos, em até 12 palavras")
    forca_hook_0a10: int = Nota
    tem_rosto: bool
    close_up_produto: bool
    before_after_ou_transformacao: bool
    try_on_ou_demonstracao: bool
    texto_na_tela: bool
    densidade_texto_0a10: int = Nota
    ritmo_cortes_0a10: int = Nota
    energia_musica_0a10: int = Nota
    tom_narracao: Literal["ausente", "calmo", "animado", "instrucional"]
    brilho_saturacao_0a10: int = Nota
    cor_dominante: str = Field(..., description="uma palavra em português")
    estetica_geral_0a10: int = Nota
    clareza_beneficio_produto_0a10: int = Nota
    elementos_video_top5: list[str] = Field(..., max_length=5)
    elementos_audio_top5: list[str] = Field(..., max_length=5)
    potencial_viral_0a10: int = Nota
    # adaptação por setor (falso quando não se aplica)
    beleza_swatch_ou_aplicacao_na_pele: bool
    moda_corpo_inteiro: bool
    auto_close_do_carro: bool
    auto_interior_ou_pov_direcao: bool
    auto_som_do_motor: bool


class Driver(BaseModel):
    factor: str
    direction: Literal["aumenta", "reduz"]
    weight: float = Field(..., description="contribuição em escala log (SHAP)")


class Comparable(BaseModel):
    post_id: str
    url: str
    title: str
    likes: int
    relative_to_channel: float
    similarity: float


class PredictionResponse(BaseModel):
    predicted_metric: float = Field(..., description="curtidas previstas em ~60 dias")
    interval_low: float
    interval_high: float
    confidence_flag: bool = Field(..., description="True = baixa confiança")
    confidence_reasons: list[str]
    relative_to_channel: float = Field(..., description="previsto / mediana dos 20 posts anteriores do canal")
    top_drivers: list[Driver]
    explanation: str
    comparables: list[Comparable]
    features_used: dict[str, float | str | None]
    model_version: str
    warnings: list[str] = []


class RankItem(BaseModel):
    posicao: int
    indice: int = Field(..., description="posição do criativo na lista de entrada")
    creative: str
    predicted_metric: float
    interval_low: float
    interval_high: float
    relative_to_channel: float


class RankResponse(BaseModel):
    ranking: list[RankItem]
    prob_primeiro_supera_segundo: float | None = Field(
        None, description="P(1º colocado ter mais curtidas que o 2º), pela incerteza dos intervalos")
    previsoes: list[PredictionResponse]


class WhatIfItem(BaseModel):
    variacao: dict[str, str]
    predicted_metric: float
    delta_vs_original_pct: float
