# Data card — Base de Shorts Galaxies v1 (rascunho)

> Status: **rascunho (G0)**. Os números vêm de `data/processed/funil_v1.json` e de `data/canais.csv`. A recoleta das estatísticas antes do treino final (PLANO F4.6) gera a v2.

## Metadados

| Campo | Valor |
|---|---|
| Responsável | Fernanda Helena L. Bezerra (Inteli Academy), case Galaxies |
| Fonte | YouTube Data API v3 (chave própria, cota de 10.000 unidades/dia) |
| Endpoints | `channels.list` (forHandle/id), `search.list` (só para descobrir canais sem handle), `playlistItems.list` (playlist de uploads `UU…`), `videos.list` (`snippet,contentDetails,statistics,status,player,liveStreamingDetails`) |
| Datas de coleta | {COLETA} (snapshot imutável em `data/raw/coleta=AAAA-MM-DD/`) |
| Janela dos vídeos | últimos 24 meses antes da coleta → {PERIODO} |
| Volume final | {VIDEOS} Shorts de {CANAIS} canais |
| Setores | {SETORES} |
| Heurística de Shorts | vertical (`player.embedHeight > embedWidth`) **e** duração ≤ 60 s (publicados antes de 15/10/2024) ou ≤ 180 s (depois). Sem dimensões do player → excluído |
| Filtro de idade | ≥ 30 dias entre a publicação e a coleta (métricas estabilizadas) |
| Alvo | `metric_target` = curtidas (`likeCount`). Curtidas ocultas viram ausentes e o vídeo sai da base; nunca são tratadas como zero. Views foram descartadas como alvo porque a contagem de Shorts mudou em 31/03/2025 e em 24/08/2026 |

## Funil

{FUNIL}

## Seleção de canais

- **Planilha de curadoria:** `data/canais.csv`, versionada. Cada linha tem handle, setor, tipo (marca ou criador), país/idioma, o canal resolvido pela API, inscritos e a decisão de incluir, com o motivo.
- **Candidatos:** 147 canais, montados antes da coleta. São marcas oficiais e criadores dos 3 setores, priorizando o Brasil e completando com marcas e criadores globais para dar volume.
- **Exclusão manual (11):** handle ou busca que resolveram o canal errado (homônimos, canais falsos), canais suspeitos e duplicatas. O motivo está na planilha.
- **Exclusão automática:** menos de 30 Shorts elegíveis. Abaixo disso, o histórico do canal é curto demais para o alvo relativo ao canal.
- **Mix:** {MIX}
- **Canais de demo:** 3 por setor, sorteados com `seed=42` entre os canais incluídos (`papel=demo`). Ficam **inteiramente fora do treino** e servem para o de-para da demonstração.

## Esquema (`data/processed/base_v1.parquet`)

| Coluna | Tipo | Descrição | Disponível antes da publicação? |
|---|---|---|---|
| `post_id` | str | ID do vídeo | sim |
| `channel_id` | str | ID do canal | sim |
| `published_at` | datetime UTC | data e hora de publicação | sim (data planejada) |
| `caption` | str | título + `\n` + descrição | sim |
| `media_ref` | str | URL do Short | sim (o próprio criativo) |
| `metric_target` | int | **curtidas na data da coleta (alvo)** | **não** |
| `channel_stats_at_collection` | int | inscritos do canal **no dia da coleta** | sim, com ressalva: é o valor de hoje, não o da data do post |
| `sector` | str | moda / beleza / automotivo | sim |
| `title`, `description` | str | partes da legenda | sim |
| `duration_s` | float | duração em segundos | sim |
| `thumbnail_url` | str | melhor thumbnail disponível | sim |
| `collected_at` | datetime UTC | momento da coleta | — (controle de idade) |
| `channel_videos_at_collection` | int | nº de vídeos do canal na coleta | sim, com a mesma ressalva |
| `channel_type`, `channel_lang` | str | marca/criador; país/idioma (curadoria) | sim |
| `role` | str | `treino` ou `demo` | — |
| `view_count_pos`, `comment_count_pos` | int | views e comentários **pós-publicação** | **não. Só para análise, nunca como feature** |

## Termos de uso e ética

- Os dados vêm só da API oficial, dentro da cota padrão. Não houve scraping, não foram criados vários projetos para somar cota e `search.list` não foi usado para listar vídeos.
- O uso é educacional e de pesquisa, no âmbito da liga Inteli Academy.
- Os dados são públicos e de canais (marcas e criadores). A base não contém dados pessoais de espectadores; não coletamos comentários nem perfis.
- Pelas políticas da API, os dados armazenados devem ser atualizados: haverá recoleta das estatísticas antes do treino final, e nenhuma coleta é mantida mais de 30 dias sem refresh.
- **Não redistribuímos** mídia (vídeos, thumbnails, frames) nem a base bruta: `data/raw`, `data/processed` e `data/media` ficam fora do git. O repositório traz os scripts para regenerar tudo com uma chave própria.

## O que esta base não representa

- **Outras plataformas** (TikTok, Reels) nem vídeos longos do YouTube.
- **Anúncios pagos:** são posts orgânicos. Não sabemos se algum post recebeu impulsionamento, o que infla curtidas sem relação com o conteúdo.
- **Canais pequenos e iniciantes:** a curadoria privilegia canais conhecidos, e o filtro de ≥30 Shorts elimina canais pouco ativos.
- **O público inteiro:** curtidas medem a reação de quem curte, não de quem assiste.
- **O período antes da janela de 24 meses** nem vídeos com menos de 30 dias.
- **Todos os idiomas e mercados:** há só português e inglês, em sua maioria.

## Vieses conhecidos

- **Snapshot único:** vídeos mais antigos tiveram mais tempo para acumular curtidas. Mitigamos com o filtro de ≥30 dias, a idade como variável de controle e o alvo relativo ao canal.
- **Inscritos "de hoje":** `channel_stats_at_collection` é o valor da coleta, não o da data do post. Por isso as features de canal priorizam o histórico de curtidas dos posts anteriores, que é correto no tempo.
- **Viés de seleção da curadoria:** marcas famosas e criadores grandes estão sobrerrepresentados. {VIES_TAMANHO}
- **Heurística de Shorts:** pode incluir vídeos verticais curtos que não foram publicados como Short, ou excluir Shorts sem dimensões do player.
- **Mistura de mercados:** canais globais (inglês) e brasileiros têm escalas de engajamento diferentes. O idioma entra como feature e o alvo relativo reduz o efeito.
