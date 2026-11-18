"""Italian delivery guards with real preservation helpers and no live model."""
from types import SimpleNamespace
import time

import pytest

from cobol_rag import query as legacy
from cobol_rag.config import AppConfig
from cobol_rag.investigation import Budget
from cobol_rag.language_delivery import deliver_italian, ITALIAN_FAILURE, ITALIAN_LOCALIZATION_FAILURE
from cobol_rag.query_plan import detect_message_language, resolve_response_language


class Replies:
    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.calls = []

    def call(self, instructions, payload, **kwargs):
        self.calls.append((instructions, payload, kwargs))
        response = next(self.replies)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize('question', [
    'Qual è lo scopo principale di DEMO?',
    'Quante variabili sono analizzate in DEMO?',
    'Mostra il codice di SHOW-ERROR in DEMO.CBL.',
    'Salve, tu parli italiano?',
    'Elenca i parametri della chiamata.',
    'Spiegami la differenza tra i programmi.',
    'Cosa significa il valore "show the source" in SHOW-ERROR?',
])
def test_italian_detection_ignores_identifiers_and_quoted_english(question):
    assert detect_message_language(question) == 'it'
    assert resolve_response_language(question)[0] == 'it'


@pytest.mark.parametrize('question', [
    'What does DEMO do?', 'Which programs call DEMO?',
    'Show the source of DEMO.CBL.', 'How many variables are recorded?',
])
def test_english_questions_remain_english(question):
    assert resolve_response_language(question)[0] == 'en'


def test_explicit_output_language_wins_over_input_language():
    assert resolve_response_language('Spiega DEMO. Answer in English.')[0] == 'en'
    assert resolve_response_language('Explain DEMO. Rispondi in italiano.')[0] == 'it'


def test_already_italian_does_not_consume_translation_or_fall_back():
    budget = Replies()
    answer = 'Sì, parlo italiano. Come posso aiutarti?'
    result, status = deliver_italian(answer, budget, legacy)
    assert result == answer and status['output'] == 'already_italian'
    assert status['passed'] and not budget.calls


def test_failure_notices_are_italian_even_when_budget_is_exhausted():
    result, status = deliver_italian('I could not verify a complete answer.', Replies(), legacy, failed=True)
    assert result == ITALIAN_FAILURE and status['output'] == 'localized_failure_notice'


def test_translation_preserves_identifiers_values_and_citations():
    answer = "The program DEMO calls 'SERV' with 400 bytes [Source 1]."
    budget = Replies([{'lines': ['Il programma [[P0]] chiama [[P1]] con [[P2]] byte [[P3]].']}])
    result, status = deliver_italian(answer, budget, legacy)
    assert result == "Il programma DEMO chiama 'SERV' con 400 byte [Source 1]."
    assert status['passed'] and len(budget.calls) == 1
    assert not legacy._translation_validation_reasons(answer, result, preserve_layout=True)


def test_one_targeted_repair_for_missing_immutable_token():
    answer = 'The program DEMO returns 0.'
    budget = Replies([{'lines': ['Il programma [[P0]] restituisce un valore.']},
                      {'lines': ['Il programma [[P0]] restituisce [[P1]].']}])
    result, status = deliver_italian(answer, budget, legacy)
    assert result == 'Il programma DEMO restituisce 0.'
    assert status['output'] == 'translated_after_repair'
    assert len(budget.calls) == 2
    assert 'invalid_line_or_placeholder_structure' in budget.calls[1][0]


@pytest.mark.parametrize('bad', [
    {'lines': []}, {'lines': ['one', 'two']}, {'lines': [None]},
    {'lines': ['Il programma [[P0]] restituisce [[P1]].\nExtra']},
    {'lines': ['The program [[P0]] returns [[P1]].']},
])
def test_repeated_bad_localization_never_delivers_english_or_corrupted_answer(bad):
    budget = Replies([bad, bad])
    result, status = deliver_italian('The program DEMO returns 0.', budget, legacy)
    assert result == ITALIAN_LOCALIZATION_FAILURE
    assert not status['passed'] and len(status['attempts']) == 2


def test_transport_failure_has_bounded_italian_notice():
    budget = Replies([RuntimeError('deadline exceeded'), RuntimeError('deadline exceeded')])
    result, status = deliver_italian('The program returns a value.', budget, legacy)
    assert result == ITALIAN_LOCALIZATION_FAILURE and not status['passed']
    assert len(budget.calls) == 2


@pytest.mark.parametrize('code', [
    "```cobol\n       DISPLAY 'The record is missing'.\n       CALL WNOMEPGM USING AREA.\n```",
    "~~~jcl\n//RUN EXEC PGM=DEMO,PARM='2023,11'\n~~~",
    "       CALL WNOMEPGM USING AREA.\n       DISPLAY 'The record is missing'.",
])
def test_code_is_restored_byte_for_byte_and_only_intro_is_translated(code):
    answer = 'The source is:\n\n' + code
    budget = Replies([{'lines': ['Il codice sorgente è:']}])
    result, status = deliver_italian(answer, budget, legacy)
    assert status['passed'], status
    assert result == 'Il codice sorgente è:\n\n' + code


def test_code_only_response_needs_no_generation():
    answer = "```cobol\nDISPLAY 'Hello'.\n```"
    budget = Replies()
    result, status = deliver_italian(answer, budget, legacy)
    assert result == answer and status['output'] == 'no_prose_to_translate'
    assert not budget.calls


def test_filenames_and_source_citation_ranges_are_protected():
    answer = 'The source file is architecture.call_parameters.json [Source 2].'
    budget = Replies([{'lines': ['Il file sorgente è [[P0]] [[P1]].']}])
    result, status = deliver_italian(answer, budget, legacy)
    assert status['passed']
    assert result == 'Il file sorgente è architecture.call_parameters.json [Source 2].'
    lines, templates = legacy._separate_translation_lines('The values are recorded [E31-E34].')
    assert legacy._restore_translation_lines(['I valori sono registrati [[P0]].'], templates) == 'I valori sono registrati [E31-E34].'


def test_italian_budget_reserves_initial_translation_plus_one_output_repair():
    standard = Budget(AppConfig())
    italian = Budget(AppConfig(), italian=True)
    assert italian.maximum == standard.maximum + 3
    assert italian.output_reserve == 2
    assert italian.deadline - standard.deadline < 0.1


@pytest.fixture
def adapter(monkeypatch):
    from cobol_rag import investigation
    class AdapterBudget(Replies):
        def __init__(self, replies):
            super().__init__(replies)
            self.deadline = time.monotonic() + 240
            self.count = 0

        @property
        def calls(self):
            return self.count

        @calls.setter
        def calls(self, value):
            self.count = 0

        def call(self, instructions, payload, **kwargs):
            self.count += 1
            response = next(self.replies)
            if isinstance(response, Exception):
                raise response
            return response

    result = dict(answer='The program DEMO returns 0.', status='complete', mode='technical',
                  tools=SimpleNamespace(evidence={}, memory={}), evidence_ids=[], trace=[],
                  errors=[], requirements=['Explain DEMO'], tool_calls=0)
    monkeypatch.setattr(investigation, 'resolve_request', lambda *a: dict(
        resolved_question='Explain DEMO.', output='other', limit=None))
    monkeypatch.setattr(investigation, 'investigate', lambda *a: result)
    monkeypatch.setattr(legacy, 'write_answer_trace', lambda *a: 'test-trace')
    monkeypatch.setattr(legacy, '_trace_payload', lambda **kw: kw)
    def run(replies, question='Spiega DEMO.'):
        budget = AdapterBudget(replies)
        monkeypatch.setattr(investigation, 'Budget', lambda *a, **kw: budget)
        return investigation.answer_with_investigation(question, AppConfig()), budget
    return run, result


def test_adapter_does_not_promote_localization_failure_to_verified_answer(adapter):
    run, result = adapter
    answer, budget = run([{'text': 'Explain DEMO.'}, {'lines': []}, {'lines': []}])
    assert answer.answer == ITALIAN_LOCALIZATION_FAILURE
    assert answer.route == 'unclear' and answer.execution_mode == 'clarification'
    assert answer.debug['evidence_validation']['passed'] is True
    assert answer.debug['validation']['passed'] is False
    assert answer.debug['validation']['stage'] == 'language_delivery'
    assert answer.debug['answer_status'] == 'failed'
    assert answer.plan.investigation_memory is None
    assert budget.calls == 3


def test_adapter_localizes_failed_investigation_without_an_extra_model_call(adapter):
    run, result = adapter
    result.update(status='failed', mode='clarification', errors=['budget_exhausted'])
    answer, budget = run([{'text': 'Explain DEMO.'}])
    assert answer.answer == ITALIAN_FAILURE and budget.calls == 1
    assert answer.debug['validation']['passed'] is False
    assert answer.debug['language_adapter']['passed'] is True


def test_adapter_preserves_verified_status_on_successful_localization(adapter):
    run, _ = adapter
    answer, budget = run([{'text': 'Explain DEMO.'},
                         {'lines': ['Il programma [[P0]] restituisce [[P1]].']}])
    assert answer.answer == 'Il programma DEMO restituisce 0.'
    assert answer.debug['validation']['passed'] is True
    assert answer.plan.response_language == 'it'
    assert budget.calls == 2


def test_adapter_english_reply_needs_no_translation(adapter):
    run, _ = adapter
    answer, budget = run([], question='Explain DEMO.')
    assert answer.answer == 'The program DEMO returns 0.'
    assert answer.plan.response_language == 'en' and budget.calls == 0
