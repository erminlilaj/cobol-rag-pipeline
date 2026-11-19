"""Bounded, evidence-driven recovery; no business-question routing or invented facts."""
from collections import defaultdict
import json

from cobol_rag.investigation_tools import digest


def source_row(row):
    return (row.get('_artifact') != 'source.outline' and row.get('source_file')
            and type(row.get('line')) is int and isinstance(row.get('text'), str))


def evidence_progress(tools):
    """Measure facts rather than transport pages or freshly allocated handles."""
    keys = set()
    for row in tools.evidence.values():
        if row.get('_artifact') == 'source.outline':
            fact = {key: row.get(key) for key in ('program', 'source_file', 'name', 'first_line', 'last_line')}
        elif row.get('_artifact') == 'verified_collection_operation':
            def scope(recipe):
                if not isinstance(recipe, dict):
                    return recipe
                return {key: scope(value) if isinstance(value, dict) else value
                        for key, value in recipe.items() if key not in {'offset', 'limit', 'order_by'}}
            fact = {'recipe': scope(row.get('recipe', {})), 'total_matches': row.get('total_matches')}
        else:
            fact = {key: value for key, value in row.items()
                    if key not in {'evidence_id', 'result_id', '_row_id', 'citation'}}
        keys.add(digest(fact))
    return frozenset(keys)


def recovery_source_evidence(tools, max_chars=20000):
    """Keep actual facts despite repeated empty lookups rotating observations.

    Round-robin across programs avoids one large source window crowding out a
    second program. IDs always refer to registered evidence, never summaries.
    """
    groups = defaultdict(list)
    for eid, row in tools.evidence.items():
        if source_row(row):
            groups[row.get('program')].append({'evidence_id': eid, **{key: row[key] for key in
                ('program', 'source_file', 'line', 'paragraph', 'text', 'is_comment') if key in row}})
    result, size = [], 0
    for index in range(160):
        for rows in groups.values():
            if index >= len(rows):
                continue
            row = rows[index]
            cost = len(json.dumps(row, ensure_ascii=False))
            if size + cost > max_chars:
                return {'rows': result, 'truncated': True,
                        'limitation': 'Selected inspected source lines, not complete program coverage.'}
            result.append(row)
            size += cost
    return {'rows': result, 'truncated': any(len(rows) > 160 for rows in groups.values()),
            'limitation': 'Inspected source only; missing programs or lines remain unverified.'}


def outline_followups(tools, seen, maximum=2):
    """Read up to two uninspected hit contexts, never scan the entire corpus."""
    inspected = {(row.get('program'), row.get('source_file'), row.get('line'))
                 for row in tools.evidence.values() if source_row(row)}
    groups = defaultdict(list)
    for row in tools.evidence.values():
        if row.get('_artifact') == 'source.outline':
            groups[row.get('program')].append(row)
    calls = []
    for index in range(max((len(rows) for rows in groups.values()), default=0)):
        for rows in groups.values():
            if index >= len(rows):
                continue
            row = rows[index]
            first, last = row.get('first_line'), row.get('last_line')
            if type(first) is not int or type(last) is not int:
                continue
            missing = [match['line'] for match in row.get('matches', [])
                       if type(match.get('line')) is int and
                       (row.get('program'), row.get('source_file'), match['line']) not in inspected]
            if not missing:
                continue
            anchor = missing[0]
            start = first if last - first < 120 else max(first, anchor - 35)
            end = min(last, start + 119)
            call = {'tool': 'source_range', 'args': {'program': row['program'],
                    'source_file': row['source_file'], 'start': start, 'end': end}}
            if digest(call) not in seen and call not in calls:
                calls.append(call)
                if len(calls) >= maximum:
                    return calls
    return calls
