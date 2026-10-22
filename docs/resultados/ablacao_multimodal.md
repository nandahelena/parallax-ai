# Ablação multimodal

Subconjunto com mídia extraída: treino 627, val 248, teste 1079 (de 2543 no teste); VLM em 1176 vídeos.

## Blocos sobre o M2 (resíduo, cumulativo)

| modelo (cumulativo)                |   mae_log |   spearman_intra |   pairwise_intra | Δ pairwise vs M2 (IC95%)   | Δ spearman_intra vs M2 (IC95%)   |
|:-----------------------------------|----------:|-----------------:|-----------------:|:---------------------------|:---------------------------------|
| M2 (offset)                        |     0.725 |            0.132 |            0.574 | [0.000, 0.000]             | [0.000, 0.000]                   |
| M2 + reajuste (controle)           |     0.724 |            0.131 |            0.574 | [-0.001, 0.002]            | [-0.004, 0.002]                  |
| + thumbnail                        |     0.724 |            0.166 |            0.574 | [-0.005, 0.011]            | [0.009, 0.063]                   |
| + frames (SigLIP2)                 |     0.734 |            0.200 |            0.572 | [-0.012, 0.017]            | [0.020, 0.117]                   |
| + vídeo (OpenCV + cortes)          |     0.733 |            0.198 |            0.576 | [-0.010, 0.020]            | [0.014, 0.117]                   |
| + áudio (Whisper + energia)        |     0.725 |            0.218 |            0.578 | [-0.006, 0.022]            | [0.037, 0.135]                   |
| + detectores (rosto, OCR, objetos) |     0.730 |            0.191 |            0.574 | [-0.010, 0.019]            | [-0.002, 0.126]                  |
| + VLM (rubrica + clusters)         |     0.726 |            0.230 |            0.579 | [-0.005, 0.022]            | [0.040, 0.164]                   |

## Thumbnail na base inteira

|                |   mae_log |   spearman_intra |   pairwise_intra | Δ pairwise vs M2 (IC95%)   |
|:---------------|----------:|-----------------:|-----------------:|:---------------------------|
| M2             |     0.716 |            0.090 |            0.582 | [0.000, 0.000]             |
| M2 + thumbnail |     0.711 |            0.174 |            0.587 | [-0.004, 0.017]            |

**Não integrado:** nenhum bloco com IC95% do Δ par a par acima de zero; o v1 usa canal + tempo + texto.
