from dataclasses import dataclass
from typing import Any

from prompts.system import get_system_prompt
from utils.text import count_tokens


@dataclass
class MessageItem:
    role: str
    content: str
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content or "",
        }


class ContextManager:
    def __init__(self) -> None:
        self._system_prompt = get_system_prompt()
        self._messages: list[MessageItem] = []
        self._model_name = "poolside/laguna-m.1:free"

    def add_user_message(self, content: str) -> None:
        message = MessageItem(
            role="user",
            content=content,
            token_count=count_tokens(
                content,
                self._model_name,
            ),
        )

        self._messages.append(message)

    def add_assistant_message(self, content: str | None) -> None:
        message = MessageItem(
            role="assistant",
            content=content or "",
            token_count=count_tokens(
                content or "",
                self._model_name,
            ),
        )

        self._messages.append(message)

    def get_messages(self) -> list[dict[str, Any]]:
        messages = []

        # system prompt
        if self._system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self._system_prompt,
                }
            )

        # user related messages
        for item in self._messages:
            messages.append(item.to_dict())

        return messages
