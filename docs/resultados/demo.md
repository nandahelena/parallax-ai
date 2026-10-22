# Demonstração de-para — canais nunca vistos no treino

Cada canal: os 4 Shorts mais recentes tratados como candidatos não publicados; histórico enviado via `channel_stats` só com posts anteriores.

**Resumo:** 36 criativos de 9 canais · acerto par a par **58%** · cobertura do intervalo de 80%: **75%** · MAE_log 0.861

Respostas completas da tool em `demo/respostas/` (plano B da apresentação).

### Eudora (beleza) · trilha do agente: T1 extrair_atributos → T3 prever (LightGBM) → T2 comparaveis (k=[5, 5, 5, 8]) → T4 explicar (SHAP) → T5 ordenar

Pares ordenados certo: **4/6** · 1º lugar previsto é o 1º real: **sim** · P(1º > 2º) = 0.589

|   rank previsto |   rank real | titulo                                             |   previsto | intervalo   |   real | dentro   |   erro_log |
|----------------:|------------:|:---------------------------------------------------|-----------:|:------------|-------:|:---------|-----------:|
|               3 |           2 | É possível resetar o cabelo danificado em casa? De |      96.50 | 25–254      |     95 | True     |       0.02 |
|               1 |           1 | Chega de perder tempo com misturas caseiras: recup |     183.10 | 31–402      |    152 | True     |       0.19 |
|               4 |           3 | E se você pudesse resetar anos de danos do seu cab |      81.20 | 25–238      |     32 | True     |       0.91 |
|               2 |           4 | Testamos a força do cabelo no laboratório e o resu |     133.30 | 25–333      |     20 | False    |       1.86 |

> Explicação do 1º colocado: Previsão de ~183 curtidas em 60 dias (intervalo de 80%: 31 a 402), 3,56× a mediana recente do canal. Fatores que mais pesaram: patamar recente de curtidas do canal (aumenta); duração do vídeo (aumenta); tema e conteúdo da legenda (aumenta).

### Renault Brasil (automotivo) · trilha do agente: T1 extrair_atributos → T3 prever (LightGBM) → T2 comparaveis (k=[8, 8, 8, 8]) → T4 explicar (SHAP) → T5 ordenar

Pares ordenados certo: **4/6** · 1º lugar previsto é o 1º real: **não** · P(1º > 2º) = 0.561

|   rank previsto |   rank real | titulo                                             |   previsto | intervalo   |   real | dentro   |   erro_log |
|----------------:|------------:|:---------------------------------------------------|-----------:|:------------|-------:|:---------|-----------:|
|               4 |           4 | #TwingoLiveShow começa hoje na França, se liga na  |      64.00 | 15–334      |     28 | True     |       0.81 |
|               3 |           1 | #Renault #Boreal                                   |      75.10 | 18–583      |  19549 | False    |       5.55 |
|               1 |           2 | vale a pena atender quando é o novo Renault Kwid q |     171.10 | 14–564      |     37 | True     |       1.51 |
|               2 |           3 | novo Renault Kwid: o carro com agilidade máxima.   |     125.60 | 16–574      |     34 | True     |       1.29 |

> Explicação do 1º colocado: Previsão de ~171 curtidas em 60 dias (intervalo de 80%: 14 a 564), 2,81× a mediana recente do canal. Fatores que mais pesaram: instabilidade do canal (aumenta); patamar recente de curtidas do canal (aumenta); dias desde o último post (aumenta).

### Christian Dior (moda) · trilha do agente: T1 extrair_atributos → T3 prever (LightGBM) → T2 comparaveis (k=[5, 5, 5, 8]) → T4 explicar (SHAP) → T5 ordenar

Pares ordenados certo: **2/6** · 1º lugar previsto é o 1º real: **não** · P(1º > 2º) = 0.509

|   rank previsto |   rank real | titulo                                             |   previsto | intervalo   |   real | dentro   |   erro_log |
|----------------:|------------:|:---------------------------------------------------|-----------:|:------------|-------:|:---------|-----------:|
|               3 |           3 | Getting into character. Camille Cottin as Jonathan |     584.30 | 217–2059    |    324 | True     |       0.59 |
|               1 |           4 | Getting into character. Sophie Wilde as Jonathan A |     633.30 | 232–2507    |    271 | True     |       0.85 |
|               4 |           2 | Getting into character. Drew Starkey as Jonathan A |     581.00 | 246–2427    |    343 | True     |       0.53 |
|               2 |           1 | Getting into character. Josh O’Connor as Jonathan  |     614.90 | 229–3001    |    398 | True     |       0.43 |

> Explicação do 1º colocado: Previsão de ~633 curtidas em 60 dias (intervalo de 80%: 232 a 2.507), 1,06× a mediana recente do canal. Fatores que mais pesaram: patamar recente de curtidas do canal (reduz); desempenho do post anterior (aumenta); inscritos do canal (aumenta).

### Carro Chefe (automotivo) · trilha do agente: T1 extrair_atributos → T3 prever (LightGBM) → T2 comparaveis (k=[5, 5, 5, 5]) → T4 explicar (SHAP) → T5 ordenar

Pares ordenados certo: **1/6** · 1º lugar previsto é o 1º real: **não** · P(1º > 2º) = 0.531

|   rank previsto |   rank real | titulo                              |   previsto | intervalo   |   real | dentro   |   erro_log |
|----------------:|------------:|:------------------------------------|-----------:|:------------|-------:|:---------|-----------:|
|               4 |           1 | AÍ QUALQUER CARRO FICA BOM!         |    1472.00 | 646–5546    |  29997 | False    |       3.01 |
|               1 |           4 | O VW MAIS RARO DO BRASIL: POLO GTI! |    1899.10 | 834–5049    |   1946 | True     |       0.02 |
|               3 |           3 | GR YARIS É LEGAL PRA QUEM?          |    1514.30 | 789–5423    |   7211 | False    |       1.56 |
|               2 |           2 | TERMINANDO A TRETA COM A VW         |    1752.90 | 756–5117    |  15873 | False    |       2.20 |

> Explicação do 1º colocado: Previsão de ~1.899 curtidas em 60 dias (intervalo de 80%: 834 a 5.049), 0,93× a mediana recente do canal. Fatores que mais pesaram: patamar recente de curtidas do canal (reduz); tema e conteúdo da legenda (aumenta); desempenho do post anterior (aumenta).

### Rivian (automotivo) · trilha do agente: T1 extrair_atributos → T3 prever (LightGBM) → T2 comparaveis (k=[8, 5, 5, 8]) → T4 explicar (SHAP) → T5 ordenar

Pares ordenados certo: **5/6** · 1º lugar previsto é o 1º real: **não** · P(1º > 2º) = 0.567

|   rank previsto |   rank real | titulo                      |   previsto | intervalo   |   real | dentro   |   erro_log |
|----------------:|------------:|:----------------------------|-----------:|:------------|-------:|:---------|-----------:|
|               3 |           3 | RivianOS 2 | Rivian         |     283.40 | 92–1201     |    332 | True     |       0.16 |
|               2 |           1 | Coastal Cloud | R2 | Rivian |     300.30 | 106–1107    |    448 | True     |       0.40 |
|               1 |           2 | Forest Green | R2 | Rivian  |     377.00 | 96–1243     |    426 | True     |       0.12 |
|               4 |           4 | Small Talk | Rivian         |     216.40 | 81–1106     |    142 | True     |       0.42 |

> Explicação do 1º colocado: Previsão de ~377 curtidas em 60 dias (intervalo de 80%: 96 a 1.243), 0,86× a mediana recente do canal. Fatores que mais pesaram: tema e conteúdo da legenda (aumenta); tendência recente do canal (reduz); inscritos do canal (reduz).

### Ralph Lauren (moda) · trilha do agente: T1 extrair_atributos → T3 prever (LightGBM) → T2 comparaveis (k=[5, 5, 5, 5]) → T4 explicar (SHAP) → T5 ordenar

Pares ordenados certo: **6/6** · 1º lugar previsto é o 1º real: **sim** · P(1º > 2º) = 0.532

|   rank previsto |   rank real | titulo                                          |   previsto | intervalo   |   real | dentro   |   erro_log |
|----------------:|------------:|:------------------------------------------------|-----------:|:------------|-------:|:---------|-----------:|
|               4 |           4 | RALPH LAUREN | Polo Ralph Lauren | US Open 2026 |     145.00 | 63–615      |    144 | True     |       0.01 |
|               3 |           3 | RALPH LAUREN | Polo Ralph Lauren | US Open 2026 |     145.70 | 59–690      |    148 | True     |       0.02 |
|               1 |           1 | RALPH LAUREN | Polo Ralph Lauren | US Open 2026 |     213.50 | 68–820      |    295 | True     |       0.32 |
|               2 |           2 | RALPH LAUREN | Polo Ralph Lauren | Men's Fall   |     191.70 | 71–750      |    288 | True     |       0.41 |

> Explicação do 1º colocado: Previsão de ~214 curtidas em 60 dias (intervalo de 80%: 68 a 820), 1,13× a mediana recente do canal. Fatores que mais pesaram: patamar recente de curtidas do canal (aumenta); duração do vídeo (aumenta); tendência recente do canal (reduz).

### Karen Bachini (beleza) · trilha do agente: T1 extrair_atributos → T3 prever (LightGBM) → T2 comparaveis (k=[5, 5, 5, 5]) → T4 explicar (SHAP) → T5 ordenar

Pares ordenados certo: **3/6** · 1º lugar previsto é o 1º real: **não** · P(1º > 2º) = 0.517

|   rank previsto |   rank real | titulo                                            |   previsto | intervalo   |   real | dentro   |   erro_log |
|----------------:|------------:|:--------------------------------------------------|-----------:|:------------|-------:|:---------|-----------:|
|               4 |           2 | É SORTE OU AZAR??? AHAHAAHHA - Karen Bachini      |    3059.70 | 1219–9139   |   2133 | True     |       0.36 |
|               3 |           4 | BLUSH LÍQUIDO DA BENEFIT - Karen Bachini          |    3206.80 | 1189–7484   |   1612 | True     |       0.69 |
|               2 |           1 | CAMILA PUDIM ME DEU UMA CHINELADA - Karen Bachini |    3273.80 | 1240–9103   |   6951 | True     |       0.75 |
|               1 |           3 | PÓ DE DIAMANTE DA GUERLAIN 🚨- Karen Bachini      |    3425.80 | 1258–8862   |   1788 | True     |       0.65 |

> Explicação do 1º colocado: Previsão de ~3.426 curtidas em 60 dias (intervalo de 80%: 1.258 a 8.862), 1,33× a mediana recente do canal. Fatores que mais pesaram: patamar recente de curtidas do canal (reduz); duração do vídeo (aumenta); desempenho do post anterior (aumenta).

### Jenn Im (moda) · trilha do agente: T1 extrair_atributos → T3 prever (LightGBM) → T2 comparaveis (k=[5, 5, 5, 5]) → T4 explicar (SHAP) → T5 ordenar

Pares ordenados certo: **3/6** · 1º lugar previsto é o 1º real: **não** · P(1º > 2º) = 0.513

|   rank previsto |   rank real | titulo                    |   previsto | intervalo   |   real | dentro   |   erro_log |
|----------------:|------------:|:--------------------------|-----------:|:------------|-------:|:---------|-----------:|
|               2 |           2 | riding solo               |    1731.30 | 644–4625    |   5021 | False    |       1.06 |
|               4 |           4 | K-beauty just gets it     |    1360.20 | 584–5926    |    516 | False    |       0.97 |
|               3 |           1 | What my son eats in a day |    1511.20 | 561–5112    |   5547 | False    |       1.30 |
|               1 |           3 | weird food combos         |    1797.90 | 674–6208    |   3011 | True     |       0.52 |

> Explicação do 1º colocado: Previsão de ~1.798 curtidas em 60 dias (intervalo de 80%: 674 a 6.208), 1,05× a mediana recente do canal. Fatores que mais pesaram: patamar recente de curtidas do canal (reduz); uso de caixa alta (aumenta); tendência recente do canal (reduz).

### tarte cosmetics (beleza) · trilha do agente: T1 extrair_atributos → T3 prever (LightGBM) → T2 comparaveis (k=[5, 5, 5, 5]) → T4 explicar (SHAP) → T5 ordenar

Pares ordenados certo: **3/5** · 1º lugar previsto é o 1º real: **não** · P(1º > 2º) = 0.584

|   rank previsto |   rank real | titulo                                           |   previsto | intervalo   |   real | dentro   |   erro_log |
|----------------:|------------:|:-------------------------------------------------|-----------:|:------------|-------:|:---------|-----------:|
|               3 |           2 | get the lewwwwkkkkk bc summer is NOT over 💋🌞   |       9.00 | 2–19        |     11 | True     |       0.18 |
|               4 |           3 | holding onto summer with an everyday bronze 🌞✨ |       8.80 | 3–21        |      8 | True     |       0.09 |
|               1 |           4 | she remains #1 for a reason 💋                   |      11.80 | 2–24        |      8 | True     |       0.35 |
|               2 |           1 | the complexion duo of my dreams 💭🌞             |       9.30 | 3–18        |     25 | False    |       0.93 |

> Explicação do 1º colocado: Previsão de ~12 curtidas em 60 dias (intervalo de 80%: 2 a 24), 1,74× a mediana recente do canal. Fatores que mais pesaram: patamar recente de curtidas do canal (aumenta); tema e conteúdo da legenda (aumenta); desempenho do post anterior (reduz).
