import os
from openai import AsyncOpenAI
from typing import Any
from response import TextDelta, TokenUsage

class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_completion(
        self, messages: list[dict[str, Any]], stream: bool = True
    ):
        client = self.get_client()

        kwargs = {
            "model": "poolside/laguna-m.1:free",
            "messages": messages,
            "stream": stream,
        }

        if stream:
            self._stream_response()

        else:
            await self._non_stream_response(client, kwargs)

    async def _stream_response(self, messages: list[dict[str, Any]]):
        pass

    async def _non_stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]):
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)

        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.prompt_tokens_details.cached_tokens if response.prompt_tokens_details else 0,
            )
        else:
            usage = None

        print(response)
