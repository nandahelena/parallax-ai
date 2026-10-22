# Relatório de experimento — Parallax

## Hipóteses (registradas em 15/10/2026, antes de qualquer treino)

Base: `base_v1` (coleta de 15/10/2026), só canais de treino (80). Os 9 canais de demo ficam fora de tudo.

- **H1 — o conteúdo e o contexto acrescentam algo ao histórico do canal.**
  - O LightGBM com canal + tempo + metadados + texto (M2) **supera o baseline B0** (mediana das curtidas dos 20 Shorts anteriores do canal) no teste temporal.
  - A comparação é pela acurácia par a par intra-canal.
  - Critério: Δ ≥ +2 p.p., com IC95% do Δ (bootstrap pareado por canal) acima de zero, **e** acurácia par a par ≥ 60%, o patamar que a Galaxies considera prova de conceito.
  - Também se espera MAE_log menor que o do B0.
- **H2 — a camada multimodal ajuda pouco, mas ajuda.**
  - No mesmo subconjunto de vídeos, somar os blocos visuais, de áudio e de VLM ao M2 dá Δ Spearman intra-canal > 0, com IC95% acima de zero.
  - A literatura espera de +0,02 a +0,04. Um resultado nulo também será reportado.
- **H3 — o intervalo é calibrado.**
  - O intervalo de 80% (quantis do GBM + conformal na fatia temporal de calibração) cobre entre 75% e 85% dos casos do teste no geral, e pelo menos 70% em cada setor.

**Leitura esperada:**
- O Spearman global deve ficar alto, porque é dominado pelo tamanho do canal.
- O Spearman intra-canal deve ficar bem mais baixo, já que é aí que o conteúdo aparece.
- Um intra-canal ≥ 0,9 seria sinal de vazamento.

## Setup (G1 — Opção A)

| | |
|---|---|
| **Dados** | `base_v1`, 80 canais de treino. Entram os posts com ≥3 Shorts anteriores no canal (necessário para o alvo relativo): 16.956 linhas |
| **Split temporal** | treino: 18/10/2024 a 29/04/2026 (11.869) · val: 29/04 a 09/07/2026 (2.543) · teste: 09/07 a 15/09/2026 (2.544) |
| **Split por canal** | GroupKFold com 5 dobras sobre os 80 canais |
| **Alvo** | `y_rel = log1p(curtidas) − mediana dos 20 anteriores`; a previsão é reconvertida para log1p absoluto nas métricas |
| **Baselines** | B0 mediana do canal (20 anteriores) · B1 média · B2 regressão linear sobre log(inscritos) · B3 mediana do setor |
| **Modelos** | LightGBM com parâmetros fixos e early stopping no val. **M1** = canal + tempo + duração. **M2** = M1 + texto (12 medidas da legenda + PCA-32 do embedding MiniLM multilíngue, ajustado no treino) |
| **Incerteza** | IC95% por bootstrap de **canais** (300 reamostras, pareado entre modelos) |
| **Rastreio** | MLflow `parallax-modelo-a` (`mlflow.db`); run do M2 `3e0063bcde1646a98e204dd5faba32d1`; seed 42 |

Tabela completa: [resultados/modelo_v0_primeira_rodada.md](resultados/modelo_v0_primeira_rodada.md).

## Resultados da 1ª rodada (base v1, sem calibração)

> Primeira rodada, mantida para rastreabilidade. **Os resultados finais estão na seção "Resultados finais (v1)"**, com a base v2 e o intervalo calibrado.


**Teste temporal**

| Modelo | MAE log | Spearman global | Spearman intra-canal | **Par a par intra-canal** | Δ par a par vs B0 (IC95%) |
|---|---|---|---|---|---|
| B0 mediana do canal | 0,749 | 0,865 | −0,03 | 51,0% | — |
| M1 (canal + tempo) | 0,714 | 0,888 | 0,06 | 55,3% | +4,5 p.p. [2,5; 6,5] |
| **M2 (+ texto)** | 0,717 | 0,889 | 0,09 | **57,9%** [55,1; 61,1] | **+7,0 p.p. [5,2; 8,8]** |
| M2, alvo absoluto | 0,709 | 0,891 | 0,08 | 58,1% | +7,2 p.p. [5,4; 9,1] |

**H1 — confirmada em parte.**
- O M2 ordena criativos do mesmo canal **7 p.p. melhor que o baseline**, com IC inteiramente acima de zero e acima do limiar de +2 p.p.
- Ainda **não chega a 60%**: o IC vai de 55% a 61%.
- O ganho de MAE sobre o B0 só é significativo com o alvo absoluto (Δ [−0,087; −0,003]).
- **Leitura:** o histórico do canal sozinho não sabe ordenar posts próximos no tempo, já que o B0 fica em 51%, cara ou coroa. Os sinais de dinâmica do canal e de texto já colocam o modelo acima disso.

**Otimismo do CV aleatório (medido, nunca usado como resultado)**

| M2 avaliado com | Par a par intra-canal | Spearman intra-canal |
|---|---|---|
| Teste temporal (resultado oficial) | 57,9% | 0,09 |
| Split por canal | 62,1% (B0 nas mesmas linhas: 59,1%) | 0,34 |
| **CV aleatória** | **64,3%** | **0,39** |

- A CV aleatória infla a ordenação intra-canal em **+6,4 p.p.** e o Spearman em 4×.
- Isso acontece porque o modelo vê posts vizinhos no tempo do mesmo canal no treino. Fases de alta e de baixa do canal vazam entre as dobras.
- O split por canal cobre o período inteiro de cada canal e por isso inclui a tendência de longo prazo: até o B0 chega a 59%. O ganho do M2 sobre o B0 ali é de +3 p.p.

**Outros achados**
- **Texto ajuda:** M1 → M2 dá +2,6 p.p. no par a par e +0,03 no Spearman intra-canal.
- **Inscritos importam pouco:** tirar os inscritos (snapshot "de hoje") custa só 1,3 p.p. O modelo se apoia mais no histórico de curtidas, que é correto no tempo.
- **Intervalo de 80% bruto** (quantis do GBM): cobre **74,9%** no teste (automotivo 75,7%, beleza 73,7%, moda 74,2%). Fica levemente abaixo do alvo de 75–85%, e a calibração conformal entra na F4 (H3).
- **Importância (ganho):** dominada pela dinâmica do canal (tendência recente, instabilidade, patamar, post anterior), depois duração, cadência e componentes de texto.

---

## Resultados finais (v1: base v2 + calibração conformal)

O v1 é o mesmo M2, retreinado na `base_v2` (estatísticas recoletadas em 16/10/2026) e com o intervalo calibrado por CQR na fatia de validação. Tabela completa: [resultados/modelo_v1.md](resultados/modelo_v1.md). MLflow run `2191e7af8cbe431e8d6c152776ba4f13`.

| Modelo (teste temporal, 2.543 Shorts) | MAE log | Spearman intra | **Par a par intra** | Δ vs B0 (IC95%) |
|---|---|---|---|---|
| B0 mediana do canal | 0,750 | −0,02 | 51,0% | — |
| M1 canal + tempo | 0,710 | 0,08 | 55,5% | +4,5 p.p. [3,1; 6,3] |
| **M2 + texto (v1)** | 0,716 | 0,09 | **58,2%** [55,7; 61,0] | **+7,2 p.p. [5,8; 8,9]** |
| M2, alvo absoluto | 0,708 | 0,11 | 58,5% | +7,5 p.p. [5,8; 9,5] |

| Intervalo de 80% | Geral | Automotivo | Beleza | Moda |
|---|---|---|---|---|
| Quantis brutos | 75,1% | 75,9% | 73,4% | 75,0% |
| **Com CQR** (q = +0,138) | **81,0%** | **81,7%** | **79,7%** | **80,5%** |

### Veredito das hipóteses

- **H1 — confirmada em parte.**
  - ✅ Δ de +7,2 p.p. sobre o B0, com IC95% [5,8; 8,9]: acima do limiar de +2 p.p.
  - ⚠️ A acurácia absoluta, 58,2%, fica no limite da meta de 60%, que está dentro do IC [55,7; 61,0].
  - ⚠️ MAE_log menor que o do B0 (0,716 contra 0,750), mas o Δ só é significativo no alvo absoluto.
- **H3 — confirmada.** O CQR leva a cobertura de 75,1% para **81,0%**, dentro da faixa de 75–85%, e todos os setores ficam acima de 70% (79,7% a 81,7%).
- **H2 — ver a seção de ablação multimodal.**

### Demonstração de-para (9 canais nunca vistos)

36 criativos, os 4 Shorts mais recentes de cada canal tratados como candidatos não publicados, com histórico enviado via `channel_stats`. Resultado: **58,5%** de pares ordenados certo (31/53), cobertura do intervalo de 75% e MAE_log 0,86. Coerente com o teste temporal. O melhor caso foi Ralph Lauren (6/6 pares) e o pior, Carro Chefe (1/6). Detalhes em [resultados/demo.md](resultados/demo.md).

## Análise de erros (v1)

Completa em [resultados/analise_erros.md](resultados/analise_erros.md).

- **Sem viés sistemático por fatia**: nenhum grupo com viés médio acima de 0,15 em log. O erro é de ruído, não de calibração.
- **Onde erra mais:**
  - canais com **mais de 10 milhões de inscritos** (MAE 0,92 contra 0,68 no melhor grupo; leve subestimação);
  - **moda** (0,82 contra 0,64 em beleza);
  - **canais brasileiros** (0,84 contra 0,66 dos globais).
- **Vídeos de 15–30 s** são os mais difíceis de ordenar.
- **Exemplo concreto, impulsionamento pago:** os piores erros, para os dois lados, são da GWM Brasil.
  - O modelo previu ~10,5 mil curtidas para "GWM Tank 300 Hi4-T", que teve 54.
  - Previu ~1,1 mil para "GWM Poer P30", que teve 262 mil.
  - O canal alterna posts orgânicos com posts que parecem impulsionados, e a API não informa isso.
  - É a principal fonte de erro irreducível com dados públicos, e já estava listada no data card como "o que a base não representa".
- **Exemplo concreto, viral inesperado:** um Short da adidas sobre a Copa ("going into the semis like...") teve 1,76 milhão de curtidas, contra uma previsão de ~3,3 mil. São eventos externos (um jogo, uma celebridade) que nenhum atributo do criativo captura.

## Ablação multimodal (H2)

Tabela completa: [resultados/ablacao_multimodal.md](resultados/ablacao_multimodal.md). Desenho (regra 9: um bloco por vez, mesmo subconjunto e mesma janela): o M2 é treinado **fora** do subconjunto multimodal e prevê esse subconjunto como offset; cada bloco aprende só o resíduo que o M2 não explica, com early stopping na validação. O controle "M2 + reajuste" mede o ganho espúrio de reajustar no subconjunto.

Subconjunto (extração concluída: 2.049 de 2.090 vídeos, 41 indisponíveis): 627 de treino, 248 de validação e **1.079 dos 2.543 Shorts de teste** (42% do teste). O VLM cobriu 1.176 vídeos, limitado pela cota diária gratuita.

| Blocos (cumulativo) | Spearman intra | **Δ Spearman vs M2 (IC95%)** | Par a par | Δ par a par (IC95%) |
|---|---|---|---|---|
| M2 (offset) | 0,132 | — | 57,4% | — |
| + reajuste (controle) | 0,131 | [−0,004; 0,002] | 57,4% | [−0,001; 0,002] |
| + thumbnail | 0,166 | **[0,009; 0,063]** | 57,4% | [−0,005; 0,011] |
| + frames (SigLIP2) | 0,200 | **[0,020; 0,117]** | 57,2% | [−0,012; 0,017] |
| + vídeo (OpenCV, cortes, movimento) | 0,198 | **[0,014; 0,117]** | 57,6% | [−0,010; 0,020] |
| + áudio (Whisper, energia) | 0,218 | **[0,037; 0,135]** | 57,8% | [−0,006; 0,022] |
| + detectores (rosto, OCR, objetos) | 0,191 | [−0,002; 0,126] | 57,4% | [−0,010; 0,019] |
| **+ VLM (rubrica + clusters)** | **0,230** | **[0,040; 0,164]** | 57,9% | [−0,005; 0,022] |

### Veredito da H2 — confirmada, com ressalva

- ✅ **O conteúdo do vídeo melhora a ordenação.** Com todos os blocos, o Spearman intra-canal sobe de 0,132 para **0,230** (Δ ~+0,10, IC95% [0,040; 0,164], inteiramente acima de zero). É **o dobro ou mais** dos +0,02 a +0,04 que a literatura (SMP, SMTPD) esperava do conteúdo visual.
- ✅ **O controle descarta artefato:** reajustar o M2 no mesmo subconjunto, sem nenhuma feature nova, dá Δ ≈ 0 ([−0,004; 0,002]).
- ✅ **Quase todo bloco contribui**, de forma cumulativa: thumbnail, frames (SigLIP2), métricas de vídeo e **áudio** (o maior salto isolado, +0,02 sobre o bloco anterior) têm IC acima de zero. Isso ecoa o paper de referência sobre Shorts, em que a energia do áudio é a feature nº 1 e áudio + visual (0,71) supera o visual sozinho (0,62).
- ⚠️ **O ganho não aparece na acurácia par a par** (nenhum bloco com IC acima de zero; Δ de +0,5 p.p. com IC [−0,005; 0,022]). O Spearman mede a ordenação de toda a lista do canal; o par a par é uma decisão binária, mais grosseira e com mais variância. Com 1.079 Shorts de teste, o estudo detecta ganhos de Spearman, mas não ganhos de 1 ou 2 p.p. no par a par.
- 📌 **Decisão: a camada multimodal não entra no número do v1.** Três motivos:
  1. O critério de integração declarado é a métrica de decisão do negócio, escolher entre dois criativos, que é o par a par, e ali o ganho não é significativo.
  2. **Risco de training/serving skew na thumbnail**: a imagem usada no treino é a que o YouTube serve **depois** da publicação. Um criativo ainda não publicado não tem essa imagem, então usá-la no treino e um frame qualquer na inferência violaria a regra de paridade.
  3. Custo: o bloco completo exige baixar e processar cada vídeo (~20 s por criativo), contra alguns milissegundos do modelo tabular.
- 📌 **O que entregamos mesmo assim:** a tool **extrai e expõe** todos os atributos de vídeo e áudio em `features_used` (cortes, movimento, rosto, texto na tela, objetos, energia do áudio, transcrição), para inspeção pela persona — é o que a demo mostra. O código integra automaticamente um resíduo de mídia se uma rodada futura mostrar ganho par a par significativo (`models/<versão>/residuo_midia.txt`).
- 📌 **Recomendação para a sequência:** com o ganho de Spearman já comprovado, vale repetir o estudo com a base inteira em GPU e **trocar a métrica de decisão para o Spearman intra-canal** quando o uso for ranquear uma lista de criativos (e não só comparar dois). Nesse cenário, a camada multimodal deve ser integrada.

### Por que o ganho não aparece no par a par

As features de canal dominam: só a dinâmica recente (tendência, instabilidade, patamar, cadência) responde por ~28% do ganho do modelo. O conteúdo refina a **posição relativa** dos criativos na lista do canal, mas raramente é forte o bastante para inverter a comparação entre dois criativos específicos. É o mesmo padrão dos vencedores do SMP Challenge, onde remover o embedding de vídeo piorou o MAPE de apenas 0,175 para 0,182.
