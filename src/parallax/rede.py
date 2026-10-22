"""Queda de internet não é falha do vídeo: os jobs longos pulam sem registrar e tentam de novo depois."""

MARCAS = ("getaddrinfo", "failed to resolve", "timed out", "connection", "disconnected", "temporarily unavailable")


def erro_de_rede(e: BaseException) -> bool:
    import httpx
    msg = str(e).lower()
    return isinstance(e, (httpx.TransportError, ConnectionError, TimeoutError)) or any(m in msg for m in MARCAS)
