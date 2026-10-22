# Roteiro do vídeo de entrega (6 min)

Gravação de tela com narração. Sete cenas. Os textos são para falar com naturalidade, não para decorar.

## Antes de apertar o rec

```bash
uv run streamlit run app.py
```

> **Se o `uv` não for encontrado:** com o ambiente do projeto ativo, o prefixo `uv run` é dispensável — rode o comando direto (`streamlit run app.py`, `pytest`, `python scripts/...`). Fora do ambiente, use o caminho completo: `.venv/Scripts/python.exe -m streamlit run app.py` (Windows) ou `.venv/bin/python -m ...` (Linux/macOS).

1. Deixe o app aberto em `localhost:8501` e **faça uma previsão antes de gravar** (a primeira leva ~1 min carregando os modelos; as seguintes são instantâneas).
2. Abra em abas separadas: `docs/pipeline.md`, `docs/resultados/demo.md` e `README.md`.
3. Feche notificações e deixe o navegador em tela cheia.
4. Grave em blocos: uma cena por vez, e junte depois. Errou? Repete só a cena.

---

## Cena 1 · O problema (0:00 – 0:40)

**Na tela:** README no topo, mostrando o nome e a frase de efeito.

**Você diz:**
> "A Galaxies cria personas sintéticas que assistem a criativos e dão uma opinião. O problema é que essa opinião é só qualitativa: não existe número para conferir depois se a persona acertou. O Parallax é o componente estatístico que faltava. Antes de o vídeo ser publicado, ele diz quantas curtidas esperar, com margem de erro, qual dos criativos é o mais promissor e por quê. E a regra que organiza o projeto inteiro é esta: o número vem de um modelo treinado; o LLM só orquestra e explica, nunca chuta."

**Dica:** essa última frase é a mais importante do vídeo. Fale devagar.

---

## Cena 2 · A base (0:40 – 1:20)

**Na tela:** `docs/data_card.md`, rolando pelo funil.

**Você diz:**
> "A liga não forneceu a base, então eu montei a minha pela API oficial do YouTube. Parti de 42 mil vídeos de 89 canais de moda, beleza e automotivo. Depois filtrei: só Shorts, só com 30 dias ou mais de publicados, porque antes disso as curtidas ainda estão crescendo, e descartei os vídeos com curtidas ocultas, que viraram dado ausente e nunca zero. Sobraram 19.211 Shorts. Nove canais inteiros ficaram guardados, sem o modelo nunca ver, para a demonstração do fim. O data card documenta tudo, inclusive o que a base não representa: ela não enxerga impulsionamento pago, e isso volta a aparecer mais adiante."

**Dica:** mostre a tabela do funil enquanto fala. O número caindo de 42 mil para 19 mil é visualmente convincente.

---

## Cena 3 · A sacada do modelo (1:20 – 2:20)

**Na tela:** `docs/pipeline.md`, na seção "Decisões de projeto e o porquê".

**Você diz:**
> "A primeira decisão importante foi não prever o número bruto de curtidas. Um canal com 10 milhões de inscritos sempre vai ter mais curtidas que um com 50 mil, e isso o modelo aprenderia fácil, sem me dizer nada de útil. A pergunta que interessa ao cliente é outra: este criativo é melhor ou pior que os meus próprios vídeos de costume? Então o alvo passou a ser o desempenho relativo à mediana dos 20 Shorts anteriores do mesmo canal. Isso isola o efeito do conteúdo, que é justamente o que se quer decidir.
> A métrica de sucesso segue a mesma lógica: dados dois criativos do mesmo canal, o modelo acerta qual vai render mais? O baseline, que é simplesmente chutar a mediana do canal, acerta 51% — cara ou coroa. O modelo acerta 58,2%, com 7 pontos de vantagem e intervalo de confiança inteiramente acima de zero."

**Dica:** se quiser um slide, essa é a hora: "51% → 58,2%".

---

## Cena 4 · Avaliação honesta (2:20 – 3:10)

**Na tela:** `docs/experiment_report.md`, na tabela de otimismo.

**Você diz:**
> "Essa parte é a que eu mais cuidei. Treinei com o passado e testei com o futuro, como seria na vida real. E medi quanto eu estaria me enganando se tivesse feito do jeito fácil: embaralhando os dados ao acaso, o modelo pareceria acertar 64%, seis pontos a mais. Esse ganho é ilusório, porque o modelo veria posts vizinhos no tempo do mesmo canal e aprenderia a fase em que o canal estava. Está documentado no relatório, porque medir o próprio otimismo faz parte.
> O intervalo também foi calibrado. Os quantis crus cobriam 75% dos casos quando deveriam cobrir 80%. Com calibração conformal, ficou em 81%, e entre 80 e 82% em cada setor. Ou seja, quando a ferramenta diz 'entre 500 e 2.800', essa faixa acerta 8 de cada 10 vezes."

**Dica:** não corra aqui. É o trecho que separa um case maduro de um case bonito.

---

## Cena 5 · Demonstração ao vivo (3:10 – 4:20)

**Na tela:** a interface Streamlit, com um canal de demonstração selecionado (Rivian ou Ralph Lauren funcionam bem).

**O que fazer, nesta ordem:**
1. Mostre a lateral: o canal escolhido, inscritos e quantos Shorts ele tem.
2. Aponte os 4 criativos candidatos já carregados.
3. Clique em **Prever**.
4. Mostre o ranking, depois abra **um** criativo e percorra: previsão, intervalo, fatores, comparáveis.
5. Aponte a linha verde das curtidas reais.

**Você diz:**
> "Aqui está a ferramenta rodando num canal que o modelo nunca viu. Ele recebe o histórico, as legendas planejadas e a data, e devolve a previsão de cada criativo, ordenada do mais ao menos promissor, com a probabilidade de o primeiro superar o segundo. Para cada um eu vejo o intervalo, os fatores que mais pesaram, e Shorts reais parecidos que serviram de evidência. E como esses vídeos já foram publicados, eu sei a resposta: aqui estão as curtidas reais, dentro do intervalo previsto. No conjunto dos nove canais guardados, a ferramenta acertou 58,5% dos duelos, bem próximo do que o teste indicava — o que mostra que o resultado não foi sorte do conjunto de teste."

**Dica:** no rodapé aparece a trilha do agente (`T1 → T3 → T2 → T4 → T5`). Aponte: é a prova visual de que as cinco tools rodaram.

---

## Cena 6 · A decisão sobre o vídeo (4:20 – 5:10)

**Na tela:** `docs/resultados/ablacao_multimodal.md`.

**Você diz:**
> "A camada multimodal era obrigatória, então processei 2 mil vídeos: cortes de cena, movimento, rosto, texto na tela, áudio, transcrição, e um modelo de visão preenchendo um questionário sobre cada vídeo. O resultado é interessante. O conteúdo melhora, sim, a ordenação da lista de criativos do canal: o Spearman sobe de 0,13 para 0,23, e esse ganho é estatisticamente sólido, o dobro do que a literatura da área esperava. Só que ele não aparece na métrica do duelo, que é a que decide o negócio.
> E tem um detalhe que pesou mais: o bloco de maior ganho usava a thumbnail, e a thumbnail que eu tenho na base é a que o YouTube serve depois da publicação. Um criativo que ainda não foi ao ar não tem essa imagem. Usar isso seria treinar com uma informação que não existe na hora de prever. Então eu medi, comprovei o ganho e escolhi não colocar no número. A ferramenta extrai e mostra esses atributos para a persona inspecionar, mas eles não mexem na previsão. Medir e não usar também é resultado."

**Dica:** é o ponto alto do vídeo. Se precisar cortar algo do roteiro, **não corte esta cena**.

---

## Cena 7 · Entrega, limites e próximos passos (5:10 – 6:00)

**Na tela:** o diagrama de arquitetura do `pipeline.md`, depois o terminal com o MCP Inspector ou o `11_cliente_mcp.py`.

**Você diz:**
> "A entrega obrigatória não é a interface, é a ferramenta. O fluxo está exposto como função Python e como servidor MCP, que é o padrão pelo qual a persona da Galaxies vai consumir. Aqui está um cliente real chamando o servidor em outro processo, listando as três tools e recebendo a previsão. Tem teste automatizado provando que o número que sai pela ferramenta é exatamente o número do modelo, e um validador que reprova qualquer explicação do LLM que cite um número ou um fator que não esteja no JSON.
> Sobre os limites: o modelo erra mais em canais gigantes e no setor de moda, e os piores erros vêm de posts que parecem impulsionados, algo que a API não informa. Com mais tempo, eu refaria a extração multimodal na base inteira com GPU e treinaria um modelo especialista com a base de anúncios de um cliente — o pipeline já aceita qualquer base no esquema mínimo."

**Dica:** termine mostrando o terminal com a resposta do MCP na tela. É a imagem que fica.

---

## Números para ter na ponta da língua

| | |
|---|---|
| Base | 19.211 Shorts · 89 canais · 3 setores |
| Duelo entre criativos | baseline 51,0% → **58,2%** |
| Intervalo de 80% | cobre **81,0%** |
| Demo em canais nunca vistos | **58,5%** |
| Multimodal | Spearman 0,13 → **0,23** (medido, não integrado) |
| Custo | zero, tudo em free tier e CPU |

## Se algo falhar na gravação

- A ferramenta não responde? Abra `demo/respostas/` e mostre o JSON salvo.
- A internet caiu? Desligue "analisar o vídeo" na lateral: a previsão funciona offline.
- Sem tempo? Corte a cena 2 (a base) e resuma em uma frase na cena 1.
