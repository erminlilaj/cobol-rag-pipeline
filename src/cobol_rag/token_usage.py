"""Provider-reported generation usage; never estimate tokens from characters."""
from collections.abc import Mapping


def response_tokens(response):
    raw = getattr(response, 'raw', None)
    if not isinstance(raw, Mapping):
        return None
    usage = raw.get('usage')
    usage = usage if isinstance(usage, Mapping) else {}
    pairs = [(raw.get('prompt_eval_count'), raw.get('eval_count')),
             (usage.get('prompt_tokens'), usage.get('completion_tokens'))]
    for input_tokens, output_tokens in pairs:
        if all(type(value) is int and value >= 0 for value in (input_tokens, output_tokens)):
            return {'input_tokens': input_tokens, 'output_tokens': output_tokens}
    return None


def summarize_tokens(reports, attempted_calls):
    known = [report for report in reports if report is not None]
    input_tokens = sum(report['input_tokens'] for report in known) if known else None
    output_tokens = sum(report['output_tokens'] for report in known) if known else None
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens if known else None,
            'reported_calls': len(known), 'model_calls': attempted_calls,
            'complete': bool(attempted_calls) and len(known) == attempted_calls,
            'source': 'provider_reported',
            'scope': 'All investigation model calls, including planning, review, retries and translation; excludes embeddings.'}
