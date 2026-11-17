from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import threading
import uuid

from cobol_rag.config import AppConfig
from cobol_rag.query import REJECTED_EXECUTION_MODES, QueryAnswer, answer_query
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
    status: str = 'complete'


@dataclass
class ChatSession:
    config: AppConfig
    top_k: int | None = None
    chunk_types: list[str] | None = None
    max_history: int = 3
    turns: list[ChatTurn] = field(default_factory=list)
    state: SessionState = field(default_factory=SessionState)
    generation: int = 0
    revision: int = 0
    context_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _request_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _pending: int = field(default=0, init=False, repr=False)

    def cancel(self) -> int:
        """Abandon the answer in flight so its result never reaches chat memory.

        Generation is not the answer itself but the conversation it belongs to.
        A question the user stopped waiting for still runs to completion in its
        worker thread, and recording it would leave the next question resolving
        pronouns against an answer nobody read.
        """
        with self._state_lock:
            self.generation += 1
            return self.generation

    def is_busy(self) -> bool:
        with self._state_lock:
            return self._pending > 0

    def context_snapshot(self) -> dict:
        with self._state_lock:
            return {'id': self.context_id, 'revision': self.revision,
                    'memory_enabled': self.config.investigation.enabled and self.config.investigation.memory_enabled,
                    'persistence': 'process-local; reset on API restart or session eviction'}

    def _with_context(self, answer: QueryAnswer) -> QueryAnswer:
        return replace(answer, debug={**answer.debug, 'session_context': self.context_snapshot()})

    def ask(self, message: str, target_program: str | None = None) -> QueryAnswer:
        # Capture the cancellation epoch before waiting: reset/cancel also
        # invalidates old queued follow-ups, not just the executing request.
        with self._state_lock:
            generation = self.generation
            self._pending += 1
        try:
            with self._request_lock:
                with self._state_lock:
                    if generation != self.generation:
                        return self._with_context(QueryAnswer(question=message, answer='Stopped before execution.',
                                                  sources=[], execution_mode='cancelled', route='unclear'))
                    memory_on = not self.config.investigation.enabled or self.config.investigation.memory_enabled
                    working_state = deepcopy(self.state) if memory_on else SessionState()
                    history = self._history_context()
                return self._ask_serialized(message, target_program, generation, working_state, history)
        finally:
            with self._state_lock:
                self._pending -= 1

    def _ask_serialized(self, message, target_program, generation, working_state, history):
        answer = answer_query(
            question=message,
            config=self.config,
            top_k=self.top_k,
            chunk_types=self.chunk_types,
            conversation_history=history,
            session_state=working_state,
            target_program=target_program,
        )
        with self._state_lock:
            return self._commit_answer(message, answer, generation)

    def _commit_answer(self, message, answer, generation):
        if generation != self.generation:
            # Cancelled while this ran. The answer is still returned so a caller
            # that wants it can use it, but the session is left as though the
            # question had never been asked.
            return self._with_context(replace(answer, execution_mode='cancelled'))
        accepted = (answer.route == 'technical' and answer.execution_mode not in REJECTED_EXECUTION_MODES
                    and answer.debug.get('status') != 'rejected'
                    and answer.debug.get('answer_status') not in {'failed', 'rejected', 'unresolved'}
                    and (answer.debug.get('validation') or {}).get('passed') is not False)
        status = answer.debug.get('answer_status', 'complete') if accepted else (
            'conversational' if answer.route == 'conversational' else 'unresolved')
        self.turns.append(
            ChatTurn(
                user=message,
                assistant=answer.answer,
                sources=answer.sources,
                route=answer.route,
                scope=answer.scope,
                trace_id=answer.trace_id,
                status=status,
            )
        )
        self.turns = self.turns[-self.max_history :]
        if answer.plan and answer.plan.response_language:
            self.state.response_language = answer.plan.response_language
        # A turn that refused to stand behind an answer must not become the
        # antecedent for the next one. "How many paragraphs does PDCBVC have?"
        # fails on the analyzers' paragraph-count disagreement, and "list them"
        # then inherited the failed intent and enumerated comment fragments.
        # If nothing was established, there is nothing to continue.
        memory_on = not self.config.investigation.enabled or self.config.investigation.memory_enabled
        if accepted:
            plan = answer.plan.as_dict() if answer.plan else None
            if not memory_on and plan:
                plan = {**plan, 'investigation_memory': None}
            self.state.update(
                answer.scope,
                [str(source.metadata.get("source_id", "")) for source in answer.sources],
                plan=plan,
            )
        elif self.config.investigation.enabled and memory_on and answer.route != "conversational":
            # Remember the request, never the failed answer as a fact. Do not
            # let a follow-up silently attach to the previous successful topic.
            self.state.investigation_memory['unresolved_turn'] = {
                'question': message, 'status': 'unresolved',
                'resolved_request': deepcopy(answer.debug.get('plan', {}).get('resolved_request'))}
        if not memory_on:
            self.state.investigation_memory = {}
        self.revision += 1
        return self._with_context(answer)

    def reset(self) -> None:
        with self._state_lock:
            self.generation += 1
            self.turns.clear()
            self.state.reset()
            self.revision = 0
            self.context_id = uuid.uuid4().hex

    def last_sources(self) -> list[RetrievalResult]:
        with self._state_lock:
            return list(self.turns[-1].sources) if self.turns else []

    def _history_context(self) -> str | None:
        if self.config.investigation.enabled and not self.config.investigation.memory_enabled:
            return None
        if not self.turns:
            return None

        if self.config.investigation.enabled:
            # The structured reference context is authoritative. History adds
            # user requests and statuses, not competing assistant claims.
            return json.dumps([{'question': t.user, 'status': t.status}
                               for t in self.turns[-self.max_history:]])

        technical_turns = [
            turn for turn in self.turns if turn.route == "technical"
        ][-self.max_history :]
        if not technical_turns:
            return None

        return "\n".join(
            f"Previous user question: {turn.user}"
            for turn in technical_turns
        )
