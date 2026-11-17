"""One evidence-free reference context for optional conversational follow-ups."""
from copy import deepcopy


def query_subject(recipe):
    """Extract scope and filters without losing nested selections/comparisons."""
    args = recipe.get('args', {})
    if recipe.get('tool') == 'select':
        subject = query_subject(args.get('parent', {}))
        subject['filters'] += deepcopy(args.get('where', []))
        if args.get('order_by'):
            subject['order_by'] = args['order_by']
        return subject
    if recipe.get('tool') == 'compare':
        left, right = (query_subject(args.get(side, {})) for side in ('left', 'right'))
        return {'programs': sorted(set(left['programs'] + right['programs'])),
                'table': left['table'] if left['table'] == right['table'] else None,
                'filters': [], 'operation': args.get('operation'),
                'inputs': [left, right], 'order_by': args.get('order_by', 'name')}
    return {'programs': list(args.get('programs', [])), 'table': args.get('table'),
            'filters': deepcopy(args.get('where', [])), 'order_by': args.get('order_by', 'name')}


def reference_context(memory):
    """Saved facts are references, never current-turn evidence or authority."""
    exchange = memory.get('last_exchange') or {}
    rid = memory.get('last_result_id')
    saved = memory.get('collections', {}).get(rid) or {}
    recipe = deepcopy(saved.get('recipe', {}))
    verified = {'status': exchange.get('status', 'unknown'),
                'question': exchange.get('question'),
                'resolved_request': deepcopy(exchange.get('resolved_request')),
                'coverage': deepcopy(exchange.get('coverage', [])),
                'subject': query_subject(recipe),
                'result': {'result_id': rid if saved else None, 'recipe': recipe,
                           'total': saved.get('total'),
                           'displayed_page': deepcopy(saved.get('displayed_page'))}}
    if not exchange and not saved:
        verified['status'] = 'empty'
    unresolved = memory.get('unresolved_turn')
    if unresolved:
        return {'version': 1, 'status': 'unresolved',
                'current_request': deepcopy(unresolved),
                'previous_verified_context': verified,
                'instruction': 'The latest request is unresolved. Do not silently attach a follow-up to the previous successful topic; resolve the current request or ask for clarification.'}
    return {'version': 1, **verified,
            'instruction': 'Use only to resolve references. Replay the recipe or retrieve fresh evidence before making factual claims. Preserve partial coverage and displayed-page versus full-collection scope.'}
