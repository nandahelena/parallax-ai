# Resultados G1 — Opção A

Base: base_v2 (coleta 15/10, estatísticas recoletadas em 16/10), canais de treino, posts com ≥3 anteriores no canal: 16950 linhas.
Cortes temporais: {"train": ["2024-10-18", "2026-04-29"], "val": ["2026-04-29", "2026-07-09"], "test": ["2026-07-09", "2026-09-15"]}. Linhas: treino 11865, val 2542, teste 2543. MLflow run M2: `2191e7af8cbe431e8d6c152776ba4f13`.

## Teste temporal (IC95% por bootstrap de canais, 300 reamostras)

| modelo                    |   mae_log |   mape_log |   spearman_global |   spearman_intra |   pairwise_intra |   pairwise_intra_m01 |   f1_faixas | ic_pairwise    | ic_spearman_intra   | delta_pairwise_vs_B0   | delta_mae_vs_B0   |
|:--------------------------|----------:|-----------:|------------------:|-----------------:|-----------------:|---------------------:|------------:|:---------------|:--------------------|:-----------------------|:------------------|
| B0 mediana do canal       |     0.750 |      0.137 |             0.865 |           -0.021 |            0.510 |                0.510 |       0.165 | [0.492, 0.530] | [-0.078, 0.051]     | [0.000, 0.000]         | [0.000, 0.000]    |
| B1 média do canal         |     0.771 |      0.148 |             0.874 |           -0.038 |            0.511 |                0.512 |       0.203 | [0.489, 0.536] | [-0.099, 0.031]     | [-0.006, 0.008]        | [-0.003, 0.048]   |
| B2 linear inscritos       |     1.233 |      0.266 |             0.732 |          nan     |            0.500 |                0.500 |       0.254 | [0.500, 0.500] | [nan, nan]          | [-0.030, 0.008]        | [0.351, 0.637]    |
| B3 mediana do setor       |     1.737 |      0.434 |             0.268 |          nan     |            0.500 |                0.500 |       0.237 | [0.500, 0.500] | [nan, nan]          | [-0.030, 0.008]        | [0.775, 1.289]    |
| M1 LightGBM               |     0.710 |      0.136 |             0.888 |            0.084 |            0.555 |                0.560 |       0.279 | [0.526, 0.589] | [0.028, 0.158]      | [0.031, 0.063]         | [-0.080, -0.009]  |
| M2 LightGBM               |     0.716 |      0.138 |             0.888 |            0.090 |            0.582 |                0.588 |       0.306 | [0.557, 0.610] | [0.018, 0.163]      | [0.058, 0.089]         | [-0.075, 0.000]   |
| M2 sem inscritos LightGBM |     0.718 |      0.139 |             0.888 |            0.102 |            0.576 |                0.581 |       0.297 | [0.549, 0.606] | [0.031, 0.183]      | [0.049, 0.083]         | [-0.074, 0.005]   |
| M2 LightGBM alvo absoluto |     0.708 |      0.137 |             0.892 |            0.105 |            0.585 |                0.593 |       0.309 | [0.557, 0.617] | [0.026, 0.192]      | [0.058, 0.095]         | [-0.088, -0.006]  |

## Generalização e otimismo

|                              |   mae_log |   mape_log |   spearman_global |   spearman_intra |   pairwise_intra |   pairwise_intra_m01 |
|:-----------------------------|----------:|-----------:|------------------:|-----------------:|-----------------:|---------------------:|
| M2 — split por canal         |     0.741 |      0.155 |             0.884 |            0.333 |            0.619 |                0.627 |
| M2 — CV aleatória (otimista) |     0.698 |      0.147 |             0.898 |            0.382 |            0.641 |                0.650 |
| B0 — todas as linhas         |     0.736 |      0.151 |             0.874 |            0.236 |            0.591 |                0.597 |
| M2 — teste temporal          |     0.716 |      0.138 |             0.888 |            0.090 |            0.582 |                0.588 |

## Intervalo de 80% (quantis brutos, antes do conformal)

Cobertura no teste: 0.751 (por setor: {'automotivo': 0.759, 'beleza': 0.734, 'moda': 0.75}). **Com CQR** (q = +0.138): 0.810 (por setor: {'automotivo': 0.817, 'beleza': 0.797, 'moda': 0.805}). Largura P80 no val (log): 2.552.

## Importância (ganho) — M2, top 15

|                    |   ganho |
|:-------------------|--------:|
| ch_trend           |   0.083 |
| ch_std_prev20      |   0.075 |
| ch_last            |   0.048 |
| ch_med_prev20      |   0.047 |
| ch_days_since_last |   0.026 |
| md_duration_s      |   0.025 |
| tx_pca_15          |   0.020 |
| tx_pca_26          |   0.020 |
| tx_pca_11          |   0.020 |
| tx_pca_22          |   0.019 |
| tx_pca_17          |   0.019 |
| tx_upper_ratio     |   0.018 |
| ch_n_prev          |   0.018 |
| tx_pca_03          |   0.018 |
| tx_pca_25          |   0.018 |
