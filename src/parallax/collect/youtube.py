"""Cliente mínimo da YouTube Data API v3 (PLANO.md F1).

Fluxo barato do guia da liga: channels.list → playlistItems.list → videos.list (1 unidade cada).
search.list só para descobrir canais sem handle conhecido (balde próprio de 100 chamadas/dia).
"""

from datetime import datetime

import isodate
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from parallax import config

VIDEO_PARTS = "snippet,contentDetails,statistics,status,player,liveStreamingDetails"


class QuotaExceeded(RuntimeError):
    """Cota diária esgotada: não adianta retentar, só retomar amanhã."""


def _is_transient(e: BaseException) -> bool:
    if not isinstance(e, HttpError):
        return False
    return e.resp.status >= 500 or (e.resp.status == 403 and "rateLimitExceeded" in str(e))


class YouTube:
    def __init__(self, budget: int = 9000, search_budget: int = 95):
        if not config.YOUTUBE_API_KEY:
            raise RuntimeError("YOUTUBE_API_KEY vazia no .env")
        self.api = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY, cache_discovery=False)
        self.used, self.budget = 0, budget
        self.search_used, self.search_budget = 0, search_budget

    @retry(retry=retry_if_exception(_is_transient), wait=wait_random_exponential(1, 60),
           stop=stop_after_attempt(6), reraise=True)
    def _exec(self, req, search: bool = False) -> dict:
        # Toda requisição custa cota, inclusive as que falham, então conta antes de executar.
        if search:
            if self.search_used >= self.search_budget:
                raise QuotaExceeded("orçamento diário de search.list atingido")
            self.search_used += 1
        else:
            if self.used >= self.budget:
                raise QuotaExceeded(f"orçamento local de {self.budget} unidades atingido")
            self.used += 1
        try:
            return req.execute()
        except HttpError as e:
            if "quotaExceeded" in str(e):
                raise QuotaExceeded(str(e)) from e
            raise

    def channel(self, *, handle: str | None = None, channel_id: str | None = None) -> dict | None:
        kw = {"forHandle": handle} if handle else {"id": channel_id}
        resp = self._exec(self.api.channels().list(part="snippet,statistics,contentDetails", **kw))
        items = resp.get("items", [])
        return items[0] if items else None

    def search_channel_id(self, query: str) -> str | None:
        resp = self._exec(self.api.search().list(part="snippet", q=query, type="channel", maxResults=1),
                          search=True)
        items = resp.get("items", [])
        return items[0]["id"]["channelId"] if items else None

    def upload_ids(self, playlist_id: str, since: datetime, limit: int) -> list[str]:
        """IDs da playlist de uploads (mais recentes primeiro) publicados a partir de `since`."""
        ids, token = [], None
        while len(ids) < limit:
            try:
                resp = self._exec(self.api.playlistItems().list(
                    part="contentDetails", playlistId=playlist_id, maxResults=50, pageToken=token))
            except HttpError as e:
                if e.resp.status == 404:  # canal sem uploads públicos: a playlist UU não existe
                    return ids
                raise
            for it in resp.get("items", []):
                published = it["contentDetails"].get("videoPublishedAt")
                if not published:  # vídeo privado ou removido
                    continue
                if datetime.fromisoformat(published) < since:
                    return ids
                ids.append(it["contentDetails"]["videoId"])
            token = resp.get("nextPageToken")
            if not token:
                break
        return ids[:limit]

    def videos(self, ids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(ids), 50):
            # maxHeight faz a API devolver embedWidth/embedHeight, que dão a orientação do vídeo.
            resp = self._exec(self.api.videos().list(part=VIDEO_PARTS, id=",".join(ids[i:i + 50]),
                                                     maxHeight=1280))
            out.extend(resp.get("items", []))
        return out


def channel_row(item: dict) -> dict:
    s, st = item["snippet"], item.get("statistics", {})
    return {
        "channel_id": item["id"],
        "titulo_api": s["title"],
        "custom_url": s.get("customUrl"),
        "pais_api": s.get("country"),
        "canal_criado_em": s["publishedAt"],
        "inscritos": int(st["subscriberCount"]) if "subscriberCount" in st else None,
        "n_videos": int(st.get("videoCount", 0)),
        "uploads": item["contentDetails"]["relatedPlaylists"]["uploads"],
    }


def video_row(item: dict) -> dict:
    s, cd, st = item["snippet"], item["contentDetails"], item.get("statistics", {})
    player = item.get("player", {})
    thumbs = s.get("thumbnails", {})
    best = next((thumbs[k]["url"] for k in ("maxres", "standard", "high", "medium", "default") if k in thumbs),
                None)
    # Ausente vira None (NaN), nunca zero: likes ocultos não são likes nulos (guia da liga).
    as_int = lambda k: int(st[k]) if k in st else None  # noqa: E731
    return {
        "video_id": item["id"],
        "channel_id": s["channelId"],
        "published_at": s["publishedAt"],
        "title": s.get("title", ""),
        "description": s.get("description", ""),
        "tags": "|".join(s.get("tags", [])),
        "category_id": s.get("categoryId"),
        "default_language": s.get("defaultLanguage"),
        "default_audio_language": s.get("defaultAudioLanguage"),
        "live_broadcast_content": s.get("liveBroadcastContent"),
        "was_live": "liveStreamingDetails" in item,
        "duration_s": isodate.parse_duration(cd["duration"]).total_seconds() if cd.get("duration") else None,
        "definition": cd.get("definition"),
        "made_for_kids": item.get("status", {}).get("madeForKids"),
        "embed_w": int(player["embedWidth"]) if "embedWidth" in player else None,
        "embed_h": int(player["embedHeight"]) if "embedHeight" in player else None,
        "thumbnail_url": best,
        "view_count": as_int("viewCount"),
        "like_count": as_int("likeCount"),
        "comment_count": as_int("commentCount"),
    }
