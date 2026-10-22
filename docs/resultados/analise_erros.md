# Análise de erros — modelo v1 (teste temporal, 2543 Shorts)

Erro = previsto − real em escala log1p. Viés > 0: o modelo superestima.

## Principais achados (gerados automaticamente)

- Por setor, o maior erro está em **moda** (MAE_log 0.82 vs 0.64 no melhor grupo; par a par 61%).
- Por tipo de canal, o maior erro está em **marca** (MAE_log 0.73 vs 0.69 no melhor grupo; par a par 58%).
- Por idioma do canal, o maior erro está em **BR** (MAE_log 0.84 vs 0.66 no melhor grupo; par a par 58%).
- Por inscritos, o maior erro está em **>10M** (MAE_log 0.92 vs 0.68 no melhor grupo; par a par 63%).
- Por duração, o maior erro está em **60-180s** (MAE_log 0.78 vs 0.59 no melhor grupo; par a par 59%).
- Por posts anteriores no canal, o maior erro está em **21-100** (MAE_log 0.87 vs 0.70 no melhor grupo; par a par 59%).

## Fatias

### Por setor

| setor      |        n |   viés (log) |   mae_log |   pairwise_intra |
|:-----------|---------:|-------------:|----------:|-----------------:|
| automotivo | 1347.000 |        0.012 |     0.695 |            0.581 |
| beleza     |  533.000 |        0.074 |     0.637 |            0.543 |
| moda       |  663.000 |       -0.086 |     0.823 |            0.615 |

### Por tipo de canal

| tipo de canal   |        n |   viés (log) |   mae_log |   pairwise_intra |
|:----------------|---------:|-------------:|----------:|-----------------:|
| criador         |  726.000 |        0.027 |     0.687 |            0.594 |
| marca           | 1817.000 |       -0.012 |     0.728 |            0.576 |

### Por idioma do canal

| idioma do canal   |        n |   viés (log) |   mae_log |   pairwise_intra |
|:------------------|---------:|-------------:|----------:|-----------------:|
| BR                |  785.000 |        0.013 |     0.839 |            0.583 |
| global            | 1758.000 |       -0.007 |     0.661 |            0.581 |

### Por inscritos

| inscritos   |        n |   viés (log) |   mae_log |   pairwise_intra |
|:------------|---------:|-------------:|----------:|-----------------:|
| <100k       |  195.000 |       -0.094 |     0.719 |            0.599 |
| 100k-1M     |  877.000 |        0.047 |     0.702 |            0.594 |
| 1M-10M      | 1223.000 |       -0.004 |     0.685 |            0.565 |
| >10M        |  248.000 |       -0.080 |     0.920 |            0.630 |

### Por duração

| duração   |       n |   viés (log) |   mae_log |   pairwise_intra |
|:----------|--------:|-------------:|----------:|-----------------:|
| ≤15s      | 637.000 |        0.013 |     0.594 |            0.586 |
| 15-30s    | 725.000 |       -0.059 |     0.725 |            0.525 |
| 30-60s    | 567.000 |        0.071 |     0.773 |            0.572 |
| 60-180s   | 614.000 |       -0.013 |     0.779 |            0.586 |

### Por posts anteriores no canal

| posts anteriores no canal   |        n |   viés (log) |   mae_log |   pairwise_intra |
|:----------------------------|---------:|-------------:|----------:|-----------------:|
| 21-100                      |  279.000 |        0.005 |     0.869 |            0.585 |
| >100                        | 2264.000 |       -0.001 |     0.697 |            0.581 |

## Maiores superestimativas

| title                                                        | sector     | channel_type   |   previsto |   real | link                                       |
|:-------------------------------------------------------------|:-----------|:---------------|-----------:|-------:|:-------------------------------------------|
| GWM Tank 300 Hi4-T: liberdade para encontrar novos caminhos  | automotivo | marca          |      10522 |     54 | https://www.youtube.com/shorts/0B6zobX8HkY |
| V2L na prática: como a GWM transforma carros em fontes de en | automotivo | marca          |       2686 |     31 | https://www.youtube.com/shorts/rcDzsaPg4mM |
| Ultra Sérum Niacinamida. Nova Embalagem, a Mesma Fórmula     | beleza     | marca          |        318 |      3 | https://www.youtube.com/shorts/ODlVUPUt_tk |
| Jorge Moraes revela tudo sobre o GWM WEY 07 de 517 cv        | automotivo | marca          |        571 |     14 | https://www.youtube.com/shorts/Zuy885LR8Og |
| O SUV que une elegância, potência e espaço para toda a famíl | automotivo | marca          |       1057 |     36 | https://www.youtube.com/shorts/a-xXO_hkKzA |
| Mateus EletricarBR mostra tudo da GWM no Festival Interlagos | automotivo | marca          |        785 |     27 | https://www.youtube.com/shorts/7V-wInKOPDU |
| UTme!×Bリーグ　撮影メイキング　田臥勇太選手編                | moda       | marca          |        874 |     35 | https://www.youtube.com/shorts/Qowqoo5o0Nc |
| Um dia no set com a gente é sempre um evento a parte ❤️      | moda       | marca          |        743 |     34 | https://www.youtube.com/shorts/iM6p_td-ijM |

## Maiores subestimativas (o modelo não viu o viral chegando)

| title                                                         | sector     | channel_type   |   previsto |    real | link                                       |
|:--------------------------------------------------------------|:-----------|:---------------|-----------:|--------:|:-------------------------------------------|
| GWM dominou Interlagos com inovação, tecnologia e lançamento  | automotivo | marca          |         27 |   89579 | https://www.youtube.com/shorts/OwhWazsP9_8 |
| Cenas Verão Arezzo 2027 | A Escolha e o Desejo                | moda       | marca          |          3 |    2691 | https://www.youtube.com/shorts/0mw5fEKNFuE |
| FIAT FESTIVAL 50 ANOS | MINIATURAS                            | automotivo | marca          |         96 |   56470 | https://www.youtube.com/shorts/rhpX03RqjkQ |
| going into the semis like...#worldcup #football #lamineyamal  | moda       | marca          |       3271 | 1760675 | https://www.youtube.com/shorts/Q8usb3_-bCQ |
| Entering a delicious world of bubbles🫧 Discover our charm co | moda       | marca          |         21 |    8961 | https://www.youtube.com/shorts/DEwuMtLXiy4 |
| Cenas Verão Arezzo 27 | O Café e o Ritual                     | moda       | marca          |          3 |    1059 | https://www.youtube.com/shorts/RfEXKUPDFDc |
| GWM Poer P30 Exclusive com teto solar e GWM Poer P30 Pro fei  | automotivo | marca          |       1099 |  262104 | https://www.youtube.com/shorts/MRfI7d9xSh8 |
| Curated by Spanish artist Rossy de Palma as the perfect expr  | moda       | marca          |         18 |    4212 | https://www.youtube.com/shorts/2XsI6Nl1Bvs |
