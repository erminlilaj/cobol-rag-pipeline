from types import SimpleNamespace
import json
import sys

import pytest

from cobol_rag.token_usage import response_tokens, summarize_tokens
from cobol_rag.investigation import Budget, budget_token_usage
from cobol_rag.config import AppConfig


@pytest.mark.parametrize('raw', [
    {'prompt_eval_count': 120, 'eval_count': 35},
    {'usage': {'prompt_tokens': 120, 'completion_tokens': 35}},
])
def test_provider_usage_formats(raw):
    assert response_tokens(SimpleNamespace(raw=raw)) == {'input_tokens': 120, 'output_tokens': 35}


@pytest.mark.parametrize('raw', [None, {}, {'prompt_eval_count': 10},
                                {'prompt_eval_count': True, 'eval_count': 1},
                                {'prompt_eval_count': -1, 'eval_count': 2},
                                {'usage': {'prompt_tokens': '100', 'completion_tokens': 2}}])
def test_missing_or_invalid_usage_is_not_estimated(raw):
    assert response_tokens(SimpleNamespace(raw=raw)) is None


def test_subtotals_missing_reports_and_zero_tokens():
    report = {'input_tokens': 120, 'output_tokens': 35}
    assert summarize_tokens([report, report], 2)['total_tokens'] == 310
    assert summarize_tokens([report, report], 2)['complete']
    partial = summarize_tokens([report, None], 3)
    assert partial['total_tokens'] == 155
    assert partial['reported_calls'] == 1 and not partial['complete']
    assert summarize_tokens([], 1)['total_tokens'] is None
    zero = summarize_tokens([{'input_tokens': 0, 'output_tokens': 0}], 1)
    assert zero['total_tokens'] == 0 and zero['complete']


def test_budget_counts_rejected_json_and_transport_gaps_without_cross_question_leaks(monkeypatch):
    responses = iter([
        SimpleNamespace(raw={'prompt_eval_count': 50, 'eval_count': 10},
                        message=SimpleNamespace(content='not json')),
        SimpleNamespace(raw={'usage': {'prompt_tokens': 70, 'completion_tokens': 20}},
                        message=SimpleNamespace(content=json.dumps({'passed': True}))),
        RuntimeError('transport failure'),
    ])
    def chat(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setitem(sys.modules, 'llama_index.core.llms', SimpleNamespace(ChatMessage=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'cobol_rag.index', SimpleNamespace(build_llm=lambda *a, **kw: SimpleNamespace(chat=chat)))
    budget = Budget(AppConfig())
    with pytest.raises(ValueError):
        budget.call('Review', {})
    assert budget.call('Review again', {}) == {'passed': True}
    assert budget_token_usage(budget)['total_tokens'] == 150
    assert budget_token_usage(budget)['complete']
    with pytest.raises(RuntimeError):
        budget.call('Review again', {})
    usage = budget_token_usage(budget)
    assert usage['total_tokens'] == 150 and not usage['complete']
    assert usage['reported_calls'] == 2 and usage['model_calls'] == 3
    assert budget_token_usage(Budget(AppConfig()))['total_tokens'] is None
