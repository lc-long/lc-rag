"""BEIR benchmark wrapper for Nova-RAG HybridRetriever.

Wraps the existing HybridRetriever to satisfy BEIR's BaseSearch interface,
allowing standardized IR benchmarking on BEIR datasets.
"""
import asyncio
import logging
from typing import Optional

from beir.retrieval.search.base import BaseSearch

from .hybrid_search import HybridRetriever
from .bm25_index import BM25Indexer

logger = logging.getLogger("nova_rag")


class NovaRAGRetriever(BaseSearch):
    """Wraps HybridRetriever for BEIR-compatible retrieval.

    BEIR's BaseSearch expects:
        search(corpus, queries, top_k) -> dict[query_id, dict[doc_id, score]]
    """

    def __init__(
        self,
        vector_store,
        embedder,
        bm25_indexer: Optional[BM25Indexer] = None,
        rrf_k: int = 30,
        distance_threshold: float = 10.0,
    ):
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            bm25_indexer=bm25_indexer,
            rrf_k=rrf_k,
            distance_threshold=distance_threshold,
        )
        self._embedder = embedder

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int = 10,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Synchronous search required by BEIR - we run async within."""
        return asyncio.run(self._async_search(corpus, queries, top_k))

    async def _async_search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int = 10,
    ) -> dict[str, dict[str, float]]:
        """Async search using HybridRetriever.

        corpus: dict[doc_id, {"title": str, "text": str, ...}]
        queries: dict[query_id, query_text]
        Returns: dict[query_id, dict[doc_id, score]]
        """
        results: dict[str, dict[str, float]] = {}

        for query_id, query_text in queries.items():
            retrieved_chunks = await self.hybrid_retriever.retrieve(query_text, top_k=top_k)

            doc_scores: dict[str, float] = {}
            for chunk in retrieved_chunks:
                doc_id = chunk.get("doc_id", "")
                rerank_score = chunk.get("rerank_score", chunk.get("distance", 0.0))
                if doc_id and doc_id not in doc_scores:
                    doc_scores[doc_id] = rerank_score
                elif doc_id and rerank_score > doc_scores[doc_id]:
                    doc_scores[doc_id] = rerank_score

            results[query_id] = doc_scores

        return results

    def encode(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        encode_output_path: str = "./embeddings/",
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """BEIR encode hook - not needed for our approach since we embed on demand."""
        pass

    def search_from_files(
        self,
        query_embeddings_file: str,
        corpus_embeddings_files: list[str],
        top_k: int,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """BEIR file-based search - not used in our approach."""
        return {}