# Resultados G1 — Opção A

Base: base_v1 (coleta=2026-10-15), canais de treino, posts com ≥3 anteriores no canal: 16956 linhas.
Cortes temporais: {"train": ["2024-10-18", "2026-04-29"], "val": ["2026-04-29", "2026-07-09"], "test": ["2026-07-09", "2026-09-15"]}. Linhas: treino 11869, val 2543, teste 2544. MLflow run M2: `3e0063bcde1646a98e204dd5faba32d1`.

## Teste temporal (IC95% por bootstrap de canais, 300 reamostras)

| modelo                    |   mae_log |   mape_log |   spearman_global |   spearman_intra |   pairwise_intra |   pairwise_intra_m01 |   f1_faixas | ic_pairwise    | ic_spearman_intra   | delta_pairwise_vs_B0   | delta_mae_vs_B0   |
|:--------------------------|----------:|-----------:|------------------:|-----------------:|-----------------:|---------------------:|------------:|:---------------|:--------------------|:-----------------------|:------------------|
| B0 mediana do canal       |     0.749 |      0.138 |             0.865 |           -0.026 |            0.510 |                0.510 |       0.166 | [0.492, 0.530] | [-0.080, 0.043]     | [0.000, 0.000]         | [0.000, 0.000]    |
| B1 média do canal         |     0.771 |      0.148 |             0.875 |           -0.041 |            0.511 |                0.512 |       0.202 | [0.489, 0.536] | [-0.102, 0.028]     | [-0.006, 0.008]        | [-0.003, 0.049]   |
| B2 linear inscritos       |     1.233 |      0.266 |             0.732 |          nan     |            0.500 |                0.500 |       0.253 | [0.500, 0.500] | [nan, nan]          | [-0.030, 0.008]        | [0.351, 0.637]    |
| B3 mediana do setor       |     1.737 |      0.435 |             0.269 |          nan     |            0.500 |                0.500 |       0.238 | [0.500, 0.500] | [nan, nan]          | [-0.030, 0.008]        | [0.775, 1.290]    |
| M1 LightGBM               |     0.714 |      0.139 |             0.888 |            0.064 |            0.553 |                0.557 |       0.299 | [0.521, 0.591] | [0.001, 0.137]      | [0.025, 0.065]         | [-0.073, -0.005]  |
| M2 LightGBM               |     0.717 |      0.140 |             0.889 |            0.094 |            0.579 |                0.585 |       0.304 | [0.551, 0.611] | [0.017, 0.172]      | [0.052, 0.088]         | [-0.073, 0.001]   |
| M2 sem inscritos LightGBM |     0.726 |      0.141 |             0.885 |            0.094 |            0.566 |                0.572 |       0.293 | [0.539, 0.600] | [0.016, 0.168]      | [0.041, 0.077]         | [-0.065, 0.009]   |
| M2 LightGBM alvo absoluto |     0.709 |      0.137 |             0.891 |            0.081 |            0.581 |                0.587 |       0.308 | [0.551, 0.614] | [0.003, 0.171]      | [0.054, 0.091]         | [-0.087, -0.003]  |

## Generalização e otimismo

|                              |   mae_log |   mape_log |   spearman_global |   spearman_intra |   pairwise_intra |   pairwise_intra_m01 |
|:-----------------------------|----------:|-----------:|------------------:|-----------------:|-----------------:|---------------------:|
| M2 — split por canal         |     0.740 |      0.155 |             0.885 |            0.338 |            0.621 |                0.629 |
| M2 — CV aleatória (otimista) |     0.693 |      0.146 |             0.900 |            0.392 |            0.643 |                0.652 |
| B0 — todas as linhas         |     0.736 |      0.151 |             0.874 |            0.236 |            0.591 |                0.597 |
| M2 — teste temporal          |     0.717 |      0.140 |             0.889 |            0.094 |            0.579 |                0.585 |

## Intervalo de 80% (quantis brutos, antes do conformal)

Cobertura no teste: 0.749 (por setor: {'automotivo': 0.757, 'beleza': 0.737, 'moda': 0.742}). Largura P80 no val (log): 2.286.

## Importância (ganho) — M2, top 15

|                    |   ganho |
|:-------------------|--------:|
| ch_trend           |   0.082 |
| ch_std_prev20      |   0.071 |
| ch_med_prev20      |   0.049 |
| ch_last            |   0.048 |
| md_duration_s      |   0.026 |
| ch_days_since_last |   0.025 |
| tx_pca_11          |   0.023 |
| tx_pca_26          |   0.021 |
| tx_pca_15          |   0.020 |
| tx_pca_25          |   0.019 |
| ch_n_prev          |   0.019 |
| tx_pca_07          |   0.018 |
| tx_pca_22          |   0.018 |
| tx_upper_ratio     |   0.018 |
| tx_pca_20          |   0.017 |
