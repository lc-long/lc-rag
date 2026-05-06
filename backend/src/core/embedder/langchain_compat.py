"""LangChain interface wrappers for custom implementations."""
from typing import Any

from langchain_core.embeddings import Embeddings


class LangChainEmbeddings(Embeddings):
    """LangChain Embeddings wrapper for any custom embedder.

    Implements langchain_core.embeddings.Embeddings interface with
    async support via thread pool.
    """

    def __init__(self, embedder: Any):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._embedder.embed([text])
        return embeddings[0] if embeddings else []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        return await asyncio.to_thread(self._embedder.embed, texts)

    async def aembed_query(self, text: str) -> list[float]:
        import asyncio
        embeddings = await asyncio.to_thread(self._embedder.embed, [text])
        return embeddings[0] if embeddings else []
