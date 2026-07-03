import asyncio
import os
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from typing import Any, AsyncGenerator
from .response import StreamEventType, StreamEvent, TextDelta, TokenUsage


class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._max_retries: int = 3

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
    ) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()

        kwargs = {
            "model": "poolside/laguna-m.1:free",
            "messages": messages,
            "stream": stream,
        }

        for attempt in range(self._max_retries + 1):
            try:

                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event

                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
                return  # Exit the function after yielding the error event

            except RateLimitError as e:
                if attempt < self._max_retries:
                    # attempt -> failed -> wait -> retry
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Rate limit exceeded after {self._max_retries} retries: {str(e)}",
                    )
                    return  # Exit the function after yielding the error event

            except APIConnectionError as e:
                if attempt < self._max_retries:
                    # attempt -> failed -> wait -> retry
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"API connection error after {self._max_retries} retries: {str(e)}",
                    )
                    return  # Exit the function after yielding the error event

            except APIError as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"API error after {self._max_retries} retries: {str(e)}",
                )
                return  # Exit the function after yielding the error event
            except Exception as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"Unexpected error: {str(e)}",
                )
                return  # Exit the function after yielding the error event

    async def _stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None

        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=(
                        chunk.usage.prompt_tokens_details.cached_tokens
                        if chunk.usage.prompt_tokens_details
                        else 0
                    ),
                )
                yield StreamEvent(type=StreamEventType.RESPONSE, usage=usage)

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(content=delta.content),
                )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _non_stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=(
                    response.usage.prompt_tokens_details.cached_tokens
                    if response.usage.prompt_tokens_details
                    else 0
                ),
            )

        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
