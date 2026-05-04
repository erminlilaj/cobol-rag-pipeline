from __future__ import annotations

from dataclasses import dataclass, field

from cobol_rag.config import AppConfig
from cobol_rag.query import QueryAnswer, answer_query
from cobol_rag.retrieve import RetrievalResult


@dataclass(frozen=True)
class ChatTurn:
    user: str
    assistant: str
    sources: list[RetrievalResult]


@dataclass
class ChatSession:
    config: AppConfig
    top_k: int | None = None
    chunk_types: list[str] | None = None
    max_history: int = 6
    turns: list[ChatTurn] = field(default_factory=list)

    def ask(self, message: str) -> QueryAnswer:
        question = self._question_with_history(message)
        answer = answer_query(
            question=question,
            config=self.config,
            top_k=self.top_k,
            chunk_types=self.chunk_types,
        )
        self.turns.append(
            ChatTurn(
                user=message,
                assistant=answer.answer,
                sources=answer.sources,
            )
        )
        self.turns = self.turns[-self.max_history :]
        return QueryAnswer(
            question=message,
            answer=answer.answer,
            sources=answer.sources,
        )

    def reset(self) -> None:
        self.turns.clear()

    def last_sources(self) -> list[RetrievalResult]:
        if not self.turns:
            return []
        return self.turns[-1].sources

    def _question_with_history(self, message: str) -> str:
        if not self.turns:
            return message

        history = "\n".join(
            f"User: {turn.user}\nAssistant: {turn.assistant}"
            for turn in self.turns[-self.max_history :]
        )
        return "\n".join(
            [
                "Use this conversation history only to resolve follow-up references.",
                "Do not treat conversation history as indexed evidence.",
                "",
                "Conversation history:",
                history,
                "",
                "Current question:",
                message,
            ]
        )
