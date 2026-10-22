# Roteiro da demonstração (5–7 min)

> 📖 **Apoio visual:** abra [pipeline.md](pipeline.md) — tem os diagramas do fluxo offline, do fluxo online e da sequência das 5 tools, prontos para mostrar na tela.

**Antes de começar**
- Terminal aberto na pasta do projeto, `uv sync` feito, `.env` preenchido.
- MCP Inspector aberto: `npx @modelcontextprotocol/inspector uv run python src/parallax/mcp_server.py`.
- Plano B à mão: `demo/respostas/*.json`, `docs/resultados/demo.md` e o vídeo gravado.

---

### 0:00 — O problema (40 s)
> "A persona da Galaxies assiste ao criativo e dá uma opinião, mas não tem um número para conferir depois. O Parallax é o componente estatístico: antes de publicar, ele diz quantas curtidas um Short deve ter, com margem de erro, qual criativo é mais promissor e por quê."

- **Pergunta norteadora:** dá para prever, antes de publicar, como o público vai reagir, e com que margem de erro?
- **Regra de ouro:** o número vem do modelo treinado; o LLM só orquestra e explica.

### 0:40 — A base (40 s)
- **19.211 Shorts de 89 canais** de moda, beleza e automotivo, coletados pela API oficial do YouTube.
- Mostrar o funil do [data card](data_card.md): 42 mil vídeos → só Shorts → 30 dias ou mais → curtidas visíveis.
- **O que a base não representa:** posts impulsionados, outras plataformas, canais pequenos.

### 1:20 — Baseline vs. modelo (50 s)
- Métrica de decisão: **dados dois criativos do mesmo canal, o modelo acerta qual terá mais curtidas?**
- **Baseline (mediana do canal): 51%**, ou seja, cara ou coroa.
- **Parallax: 58,2%**, +7,2 p.p. com IC95% [5,8; 8,9]. O intervalo de 80% cobre 81% dos casos de teste.
- **Honestidade:** CV aleatória daria 64%, um otimismo falso. Usamos split temporal (treinar no passado, testar no futuro).

### 2:10 — Camada multimodal (50 s)
- O sistema assiste o vídeo: cortes, cor, movimento, rosto, texto na tela, áudio e uma rubrica preenchida por VLM (gancho, antes/depois, demonstração).
- Mostrar a tabela de ablação: **Spearman intra-canal sobe de 0,132 para 0,230** (Δ +0,10, IC95% [0,040; 0,164]) — o dobro do que a literatura esperava (+0,02 a +0,04). O áudio é o maior salto isolado.
- **Mas o par a par não melhora** (IC inclui zero), então a camada **não entra no número**: a tool extrai e mostra os atributos do vídeo para a persona inspecionar. Dizer que **medir e não integrar também é resultado**, e explicar o motivo (métrica de decisão + a thumbnail não existe antes de publicar).

### 3:00 — Ao vivo: 3 criativos de um canal nunca visto (90 s)
- Usar um canal de demo, por exemplo **Rivian** ou **Ralph Lauren**.
- Chamar `ranquear_criativos` com 3 Shorts recentes.
- Mostrar, nesta ordem:
  1. previsão;
  2. intervalo;
  3. ranking e P(1º > 2º);
  4. `top_drivers`;
  5. comparáveis;
  6. explicação.
- Revelar as **curtidas reais** (de-para).
- Resultado geral da demo: **58,5%** de pares certos em 9 canais nunca vistos; o intervalo cobriu 75% dos valores reais.

### 4:30 — Chamando via MCP (60 s)
- No MCP Inspector: listar as tools (`prever_criativo`, `ranquear_criativos`, `simular_what_if`) e o resource `parallax://modelo`.
- Rodar `simular_what_if`: "e se publicarmos sábado às 11h?"
- Mostrar `explicar_com_llm=true` e explicar o **validador**, que reprova qualquer número ou fator fora do JSON do modelo.

### 5:30 — Limitações e próximos passos (60 s)
- **Erra mais** em canais gigantes, moda e canais BR. Impulsionamento pago é invisível na API.
- A meta de 60% ainda não foi atingida no teste temporal.
- **Próximos passos:**
  - multimodal em GPU na base inteira;
  - treino especialista com a base de ads de cada cliente;
  - detectar posts impulsionados.

---

**Dicas**
- Grave o vídeo de backup antes: cota de API e internet podem falhar ao vivo.
- Se a chamada ao vivo falhar, abra `demo/respostas/<canal>.json` e `docs/resultados/demo.md`.
