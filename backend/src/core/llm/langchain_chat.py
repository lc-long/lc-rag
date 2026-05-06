"""LangChain BaseChatModel implementation for MiniMax."""
import asyncio
import os
import json
import logging
from collections import OrderedDict
from typing import Optional, Iterator, Any, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun

logger = logging.getLogger("nova_rag")


def _to_minimax_message(msg: BaseMessage) -> dict:
    """Convert LangChain BaseMessage to MiniMax message dict."""
    content = msg.content if hasattr(msg, "content") else str(msg)
    role = getattr(msg, "type", "user")
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    elif role == "system":
        role = "user"
    return {"role": role, "content": content}


class MinimaxChatModel(BaseChatModel):
    """LangChain BaseChatModel implementation for MiniMax with streaming support.

    Wraps the existing MinimaxClient streaming logic while implementing the
    LangChain BaseChatModel interface for LCEL integration.
    """

    api_key: Optional[str] = None
    group_id: Optional[str] = None
    model: str = "MiniMax-M2.7"
    base_url: str = "https://api.minimaxi.com/v1"
    max_context_tokens: int = 6000
    _response_cache: Optional[OrderedDict] = None
    _embedder: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _cache(self) -> OrderedDict:
        if self._response_cache is None:
            self._response_cache = OrderedDict()
        return self._response_cache

    @property
    def embedder(self):
        if self._embedder is None:
            from ..embedder.aliyun_embedder import AliyunEmbedder
            self._embedder = AliyunEmbedder()
        return self._embedder

    @property
    def client(self) -> httpx.AsyncClient:
        if not hasattr(self, "_client") or self._client.is_closed:
            import httpx
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        return self._client

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def _llm_type(self) -> str:
        return "minimax"

    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "group_id": self.group_id,
            "base_url": self.base_url,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generate - runs async stream in event loop."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _agenerate():
            chat_result = ChatResult(generations=[])
            full_content = ""

            async for chunk in self._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                if isinstance(chunk, ChatGenerationChunk):
                    full_content += chunk.text

            if full_content:
                ai_message = AIMessage(content=full_content)
                gen = ChatGeneration(message=ai_message)
                chat_result.generations.append(gen)

            return chat_result

        return loop.run_until_complete(_agenerate())

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate."""
        full_content = ""
        reasoning_content = ""

        async for chunk in self._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            if isinstance(chunk, ChatGenerationChunk):
                msg = chunk.message
                if hasattr(msg, "content"):
                    full_content += msg.content
                if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("reasoning_content"):
                    reasoning_content += msg.additional_kwargs["reasoning_content"]

        ai_message = AIMessage(content=full_content)
        if reasoning_content:
            ai_message.additional_kwargs["reasoning_content"] = reasoning_content

        gen = ChatGeneration(message=ai_message)
        return ChatResult(generations=[gen])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronous stream - delegates to async version."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async_gen = self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)

        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Async streaming implementation using MiniMax API directly."""
        minimax_messages = [_to_minimax_message(msg) for msg in messages]
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in minimax_messages])

        headers = {
            "Authorization": f"Bearer {self.api_key or os.getenv('MINIMAX_API_KEY', '')}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "group_id": self.group_id or os.getenv("MINIMAX_GROUP_ID", ""),
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        got_content = False

        async with self.client.stream(
            "POST",
            f"{self.base_url}/text/chatcompletion_v2",
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise Exception(f"MiniMax API error: {response.status_code} - {body.decode()}")

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)
                    if not isinstance(data, dict):
                        continue

                    choices = data.get("choices")
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        continue

                    reasoning = delta.get("reasoning_content", "")
                    content = delta.get("content", "")

                    if reasoning:
                        got_content = True
                        from langchain_core.messages import AIMessageChunk
                        chunk_msg = AIMessageChunk(content=reasoning)
                        chunk_msg.additional_kwargs["reasoning_content"] = reasoning
                        yield ChatGenerationChunk(message=chunk_msg)

                    if content:
                        got_content = True
                        from langchain_core.messages import AIMessageChunk
                        chunk_msg = AIMessageChunk(content=content)
                        yield ChatGenerationChunk(message=chunk_msg)

                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        return

                except json.JSONDecodeError:
                    continue

        if not got_content:
            from langchain_core.messages import AIMessageChunk
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))

    async def aclose(self):
        if hasattr(self, "_client") and not self._client.is_closed:
            await self._client.aclose()

    def __del__(self):
        if hasattr(self, "_client") and not self._client.is_closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pass
                else:
                    loop.run_until_complete(self.aclose())
            except Exception:
                pass
