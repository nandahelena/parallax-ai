# Model card — Parallax v1

| | |
|---|---|
| **Versão** | `v1`. Artefatos em `models/v1/`; MLflow `parallax-modelo-a`, run `2191e7af8cbe431e8d6c152776ba4f13` |
| **Tarefa** | Regressão: prever, antes da publicação, as curtidas de um YouTube Short em ~60 dias, relativas à mediana dos 20 Shorts anteriores do canal. Saída reconvertida para curtidas absolutas |
| **Algoritmo** | LightGBM (regressão L2) para o ponto; LightGBM quantílico (0,1 e 0,9) para o intervalo de 80%, calibrado com conformal (CQR, q = +0,138 em log) |
| **Dados** | `base_v2`: 89 canais de moda, beleza e automotivo (80 de treino), Shorts de out/2024 a set/2026, estatísticas de 16/10/2026. Ver [data card](data_card.md) |
| **Períodos** | treino 18/10/2024–29/04/2026 (11.865) · validação/calibração 29/04–09/07/2026 (2.542) · teste 09/07–15/09/2026 (2.543) |

## Features (todas disponíveis antes da publicação)

| Bloco | Nº | Origem no contrato da tool |
|---|---|---|
| Canal | 14 | `channel_id`/`channel_stats`: histórico de curtidas **só de posts anteriores** (patamar, tendência, instabilidade, cadência), inscritos, tipo, idioma, setor |
| Tempo | 5 | `planned_datetime`: hora, dia da semana, mês; idade de controle fixada em 60 dias na inferência |
| Metadados | 1 | `creative`: duração |
| Texto | 44 | `caption`: 12 medidas (hashtags, emojis, CTA, caixa alta...) + 32 componentes PCA do embedding MiniLM multilíngue |
| Mídia (camada multimodal) | 34 + 16 PCA | `creative` (vídeo): extraída e exposta em `features_used` para inspeção. **Não entra no número no v1**: melhora o Spearman intra-canal (+0,10, IC [0,040; 0,164]) mas não o par a par; ver [ablação](resultados/ablacao_multimodal.md) |

## Métricas no teste temporal

| Métrica (2.543 Shorts, 09/07–15/09/2026) | Baseline B0 (mediana do canal) | **Parallax v1** |
|---|---|---|
| Acurácia par a par intra-canal (decisão) | 51,0% | **58,2%** [55,7; 61,0] · Δ +7,2 p.p. [5,8; 8,9] |
| Spearman intra-canal | −0,02 | 0,09 |
| Spearman global | 0,865 | 0,888 |
| MAE em log | 0,750 | 0,716 |
| MAPE em log | 0,137 | 0,138 |
| F1 macro nas faixas (abaixo / média / acima / viral) | 0,17 | 0,31 |
| **Cobertura do intervalo de 80%** (CQR) | — | **81,0%** · automotivo 81,7% · beleza 79,7% · moda 80,5% |

- Split por canal ("cliente novo"): par a par 61,9%. Demo em 9 canais nunca vistos: ver [demo](resultados/demo.md).
- A CV aleatória daria 64,1%: otimismo de ~6 p.p. por misturar fases do mesmo canal; **não é usada como resultado**.
- Comparação com a literatura: Spearman 0,55–0,85 antes da publicação (SMP/SMTPD), dominado pelo tamanho do canal; o nosso global é 0,89 e o intra-canal, que é o que importa para escolher entre criativos, é bem menor.

## Uso pretendido

- Apoio à decisão, **antes da publicação**, entre criativos candidatos de um canal com histórico de Shorts: qual tende a performar melhor e em que faixa.
- Consumido como tool (função Python ou MCP) por agentes, como a persona sintética da Galaxies.

## Fora do escopo

- Canais **sem histórico** de Shorts (o modelo precisa de ≥ 1 post anterior com curtidas maduras).
- Outras plataformas (TikTok, Reels), vídeos longos, anúncios pagos ou impulsionados.
- Decisões automáticas sem revisão humana; garantia de resultado.

## Limitações e vieses

- O histórico do canal domina; o conteúdo acrescenta poucos pontos de ordenação (coerente com a literatura).
- Erro maior em canais com mais de 10 milhões de inscritos, em moda e em canais BR; posts impulsionados geram os piores erros ([análise de erros](resultados/analise_erros.md)).
- Base curada, com marcas grandes sobrerrepresentadas; moda tem poucos criadores.
- Inscritos são o valor da coleta, não o da data do post.
