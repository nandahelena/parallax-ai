"""Caminhos e constantes do projeto. Decisões documentadas em PLANO.md §5 e §10."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
CANAIS_CSV = DATA / "canais.csv"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MEDIA = DATA / "media"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

SEED = 42
SETORES = ("moda", "beleza", "automotivo")
TZ_LOCAL = "America/Sao_Paulo"

# Coleta (D02): heurística de Shorts por data e janela de coleta
SHORTS_MAX_S_ANTES = 60
SHORTS_MAX_S_DEPOIS = 180
SHORTS_REGRA_MUDOU_EM = "2024-10-15"
IDADE_MIN_DIAS = 30
JANELA_MESES = 24
MAX_SHORTS_POR_CANAL = 400
MIN_SHORTS_POR_CANAL = 30  # abaixo disso o histórico do canal é curto demais para o alvo relativo
N_DEMO_POR_SETOR = 3  # canais inteiros fora do treino, para o de-para da demo

# Modelagem (D03, D07)
K_HISTORICO = 20
MIN_HISTORICO = 3
HORIZONTE_DIAS = 60
QUANTIS = (0.1, 0.9)

# Modelos externos (D09, D13)
GEMINI_VLM = "gemini-3.5-flash-lite"
GEMINI_VLM_FALLBACK = "gemini-2.5-flash-lite"
