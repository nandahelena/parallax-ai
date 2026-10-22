# Data card — Base de Shorts Galaxies v1 (rascunho)

> Versões: **base_v1** (coleta de 15/10/2026, `data/processed/funil_v1.json`) e **base_v2** (a mesma base com curtidas e inscritos recoletados em 16/10/2026, `funil_v2.json`). O treino final usa a v2.

## Metadados

| Campo | Valor |
|---|---|
| Responsável | Fernanda Helena L. Bezerra (Inteli Academy), case Galaxies |
| Fonte | YouTube Data API v3 (chave própria, cota de 10.000 unidades/dia) |
| Endpoints | `channels.list` (forHandle/id), `search.list` (só para descobrir canais sem handle), `playlistItems.list` (playlist de uploads `UU…`), `videos.list` (`snippet,contentDetails,statistics,status,player,liveStreamingDetails`) |
| Datas de coleta | 15/10/2026: uma coleta, executada em duas rodadas no mesmo dia porque a primeira parou num canal sem uploads públicos (snapshot imutável em `data/raw/coleta=AAAA-MM-DD/`). **Recoleta das estatísticas em 16/10/2026** (`videos.list` + `channels.list`, 387 unidades): 6 Shorts sumiram (removidos/privados), nenhum ocultou curtidas, e a variação mediana das curtidas em 1 dia foi 1,00, o que confirma que 30 dias bastam para estabilizar |
| Janela dos vídeos | últimos 24 meses antes da coleta → Shorts publicados de 14/10/2024 a 15/09/2026 |
| Volume final | v1: 19.217 Shorts de 89 canais · v2: 19.211 |
| Setores | moda (6.856), beleza (4.557), automotivo (7.804) |
| Heurística de Shorts | vertical (`player.embedHeight > embedWidth`) **e** duração ≤ 60 s (publicados antes de 15/10/2024) ou ≤ 180 s (depois). Sem dimensões do player → excluído |
| Filtro de idade | ≥ 30 dias entre a publicação e a coleta (métricas estabilizadas) |
| Alvo | `metric_target` = curtidas (`likeCount`). Curtidas ocultas viram ausentes e o vídeo sai da base; nunca são tratadas como zero. Views foram descartadas como alvo porque a contagem de Shorts mudou em 31/03/2025 e em 24/08/2026 |

## Funil

| Etapa | Vídeos restantes |
|---|---|
| Coletados (uploads dos últimos 24 meses, até 1.500 por canal) | 42.024 |
| Sem live/premiere | 40.494 |
| Heurística de Shorts (vertical + duração; 0 sem dimensões) | 28.088 |
| Idade ≥ 30 dias | 26.631 |
| Curtidas visíveis (1.263 ocultas excluídas) | 25.368 |
| Sem duplicatas | 25.368 |
| No máximo 400 Shorts mais recentes por canal | 19.618 |
| Canais com ≥ 30 Shorts elegíveis | **19.217** |

| Setor | Shorts | Canais |
|---|---|---|
| Automotivo | 7.804 | 37 |
| Beleza | 4.557 | 24 |
| Moda | 6.856 | 28 |
| **Total** | **19.217** | **89** (80 treino + 9 demo; 2.021 Shorts de demo) |

Fonte: `data/processed/funil_v1.json`. Cota total da coleta: 1.976 unidades + 80 chamadas de busca.

## Seleção de canais

- **Planilha de curadoria:** `data/canais.csv`, versionada. Cada linha tem handle, setor, tipo (marca ou criador), país/idioma, o canal resolvido pela API, inscritos e a decisão de incluir, com o motivo.
- **Candidatos:** 147 canais, montados antes da coleta. São marcas oficiais e criadores dos 3 setores, priorizando o Brasil e completando com marcas e criadores globais para dar volume.
- **Exclusão manual (11):** handle ou busca que resolveram o canal errado (homônimos, canais falsos), canais suspeitos e duplicatas. O motivo está na planilha.
- **Exclusão automática:** menos de 30 Shorts elegíveis. Abaixo disso, o histórico do canal é curto demais para o alvo relativo ao canal.
- **Mix:**

89 canais incluídos de 136 resolvidos: 47 saíram por ter menos de 30 Shorts elegíveis (motivo na planilha).

| Setor | Marcas | Criadores | BR | Globais |
|---|---|---|---|---|
| Automotivo | 25 | 12 | 17 | 20 |
| Beleza | 17 | 7 | 10 | 14 |
| Moda | 26 | 2 | 7 | 21 |

Por inscritos: 11 canais com menos de 100 mil, 39 entre 100 mil e 1 milhão, 35 entre 1 e 10 milhões, e 4 acima de 10 milhões. Mediana de 181 Shorts por canal (mínimo de 34).

Canais de demo: Eudora, Tarte Cosmetics e Karen Bachini (beleza); Dior, Ralph Lauren e Clothes Encounters/Jenn Im (moda); Renault Brasil, Rivian e Carro Chefe (automotivo).
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
- Pelas políticas da API, os dados armazenados devem ser atualizados: as estatísticas foram recoletadas em 16/10/2026, antes do treino final, e nenhuma coleta fica mais de 30 dias sem refresh.
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
- **Viés de seleção da curadoria:** marcas famosas e criadores grandes estão sobrerrepresentados. Só 11 canais têm menos de 100 mil inscritos. **Moda tem só 2 criadores**: a maioria dos criadores de moda candidatos postou menos de 30 Shorts. Muitas marcas BR grandes (Riachuelo, Hering, Chevrolet, Toyota, BYD) também saíram por postarem poucos Shorts.
- **Heurística de Shorts:** pode incluir vídeos verticais curtos que não foram publicados como Short, ou excluir Shorts sem dimensões do player.
- **Mistura de mercados:** canais globais (inglês) e brasileiros têm escalas de engajamento diferentes. O idioma entra como feature e o alvo relativo reduz o efeito.
