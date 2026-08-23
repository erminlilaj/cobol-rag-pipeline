from __future__ import annotations

from dataclasses import dataclass, field

from cobol_rag.config import AppConfig
from cobol_rag.query import QueryAnswer, answer_query
from cobol_rag.retrieve import RetrievalResult
from cobol_rag.scope import QueryScope, SessionState


@dataclass(frozen=True)
class ChatTurn:
    user: str
    assistant: str
    sources: list[RetrievalResult]
    route: str = "technical"
    scope: QueryScope = field(default_factory=QueryScope)
    trace_id: str | None = None


@dataclass
class ChatSession:
    config: AppConfig
    top_k: int | None = None
    chunk_types: list[str] | None = None
    max_history: int = 3
    turns: list[ChatTurn] = field(default_factory=list)
    state: SessionState = field(default_factory=SessionState)

    def ask(self, message: str, target_program: str | None = None) -> QueryAnswer:
        history = self._history_context()
        answer = answer_query(
            question=message,
            config=self.config,
            top_k=self.top_k,
            chunk_types=self.chunk_types,
            conversation_history=history,
            session_state=self.state,
            target_program=target_program,
        )
        self.turns.append(
            ChatTurn(
                user=message,
                assistant=answer.answer,
                sources=answer.sources,
                route=answer.route,
                scope=answer.scope,
                trace_id=answer.trace_id,
            )
        )
        self.turns = self.turns[-self.max_history :]
        if answer.plan and answer.plan.response_language:
            self.state.response_language = answer.plan.response_language
        if answer.route == "technical":
            self.state.update(
                answer.scope,
                [str(source.metadata.get("source_id", "")) for source in answer.sources],
                plan=answer.plan.as_dict() if answer.plan else None,
            )
        return QueryAnswer(
            question=message,
            answer=answer.answer,
            sources=answer.sources,
            route=answer.route,
            scope=answer.scope,
            trace_id=answer.trace_id,
            guard_status=answer.guard_status,
            plan=answer.plan,
            execution_mode=answer.execution_mode,
            debug=answer.debug,
        )

    def reset(self) -> None:
        self.turns.clear()
        self.state.reset()

    def last_sources(self) -> list[RetrievalResult]:
        if not self.turns:
            return []
        return self.turns[-1].sources

    def _history_context(self) -> str | None:
        if not self.turns:
            return None

        technical_turns = [
            turn for turn in self.turns if turn.route == "technical"
        ][-self.max_history :]
        if not technical_turns:
            return None

        return "\n".join(
            f"Previous user question: {turn.user}"
            for turn in technical_turns
        )
