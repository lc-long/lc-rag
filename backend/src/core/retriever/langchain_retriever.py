"""LangChain BaseRetriever implementation wrapping HybridRetriever."""
import asyncio
from typing import Optional, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class LangChainHybridRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapping the custom HybridRetriever.

    Implements langchain_core.retrievers.BaseRetriever for LCEL integration.
    The wrapped HybridRetriever is used for the actual retrieval logic.
    """

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        hybrid_retriever: Any,
        *,
        top_k: int = 8,
        doc_id: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._retriever = hybrid_retriever
        self._top_k = top_k
        self._doc_id = doc_id
        self._doc_ids = doc_ids

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Synchronous retrieval - runs async version in event loop."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._aget_relevant_documents(query, run_manager=run_manager, **kwargs)
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Async retrieval using the wrapped HybridRetriever."""
        doc_id = kwargs.get("doc_id", self._doc_id)
        doc_ids = kwargs.get("doc_ids", self._doc_ids)
        top_k = kwargs.get("top_k", self._top_k)

        if doc_ids and len(doc_ids) == 1:
            chunks = await self._retriever.retrieve(query, top_k=top_k, doc_id=doc_ids[0])
        elif doc_ids:
            chunks = await self._retriever.retrieve_multi_docs(query, top_k=top_k, doc_ids=doc_ids)
        else:
            chunks = await self._retriever.retrieve(query, top_k=top_k, doc_id=doc_id)

        return self._chunks_to_documents(chunks)

    def _chunks_to_documents(self, chunks: list[dict]) -> list[Document]:
        """Convert internal chunk dicts to LangChain Documents."""
        documents = []
        for chunk in chunks:
            content = chunk.get("parent_content", "") or chunk.get("child_content", "")
            if not content:
                continue

            doc = Document(
                page_content=content,
                metadata={
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_type": chunk.get("_type", ""),
                    "parent_id": chunk.get("parent_id", ""),
                    "child_id": chunk.get("child_id", ""),
                    "page_number": chunk.get("page_number", 0),
                    "distance": chunk.get("distance", 0.0),
                    "bm25_score": chunk.get("bm25_score", 0.0),
                    "rerank_score": chunk.get("rerank_score", 0.0),
                    "source": (chunk.get("metadata_") or {}).get("source", ""),
                },
            )
            documents.append(doc)
        return documents

    def invoke(self, input: str, config: Optional[Any] = None, **kwargs) -> list[Document]:
        """Synchronous invoke - wraps async _aget_relevant_documents."""
        return self._get_relevant_documents(input, **kwargs)

    async def ainvoke(self, input: str, config: Optional[Any] = None, **kwargs) -> list[Document]:
        """Async invoke."""
        return await self._aget_relevant_documents(input, **kwargs)
