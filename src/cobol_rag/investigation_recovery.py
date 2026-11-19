"""Bounded, evidence-driven recovery; no business-question routing or invented facts."""
from collections import defaultdict
import json

from cobol_rag.investigation_tools import digest


def source_row(row):
    return (row.get('_artifact') != 'source.outline' and row.get('source_file')
            and type(row.get('line')) is int and isinstance(row.get('text'), str))


def source_address(row):
    return (str(row.get('program') or '').upper(), str(row.get('source_file') or '').upper(), row['line'])


def source_window_already_read(call, evidence):
    """Skip only fully covered ranges; an overlapping range can add a condition."""
    if call['tool'] != 'source_range':
        return False
    args = call['args']
    if set(args) - {'program', 'source_file', 'start', 'end'} or not isinstance(args.get('program'), str):
        return False
    if 'source_file' in args and not isinstance(args['source_file'], str):
        return False
    first, last = args.get('start'), args.get('end')
    if type(first) is not int or type(last) is not int or not 1 <= first <= last or last - first >= 500:
        return False
    program = str(args.get('program', '')).upper()
    member = str(args.get('source_file') or program + '.CBL').upper()
    covered = {source_address(row) for row in evidence.values() if source_row(row)}
    return all((program, member, line) in covered for line in range(first, last + 1))


def unresolved_source_searches(observations, evidence, programs, maximum=4):
    """Try a different lookup on scoped owners whose literal outlines found nothing."""
    inspected = {source_address(row)[0] for row in evidence.values() if source_row(row)}
    named = {program.upper() for program in programs}
    calls = []
    for observation in observations:
        call = observation.get('call', {})
        args = call.get('args', {})
        program = str(args.get('program', '')).upper()
        terms = args.get('terms') or []
        if (call.get('tool') != 'source_outline' or observation.get('result', {}).get('total_matches') != 0
                or program not in named or program in inspected or not terms):
            continue
        for term in terms[:2]:
            lookup = {'tool': 'search', 'args': {'programs': [program], 'text': term, 'mode': 'literal', 'limit': 6}}
            if term.strip() and lookup not in calls:
                calls.append(lookup)
                if len(calls) >= maximum:
                    return calls
    return calls


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
    indexed = defaultdict(dict)
    for eid, row in tools.evidence.items():
        if source_row(row):
            key = (row.get('program'), row.get('source_file'))
            indexed[key].setdefault(row['line'], {'evidence_id': eid, **{field: row[field] for field in
                ('program', 'source_file', 'line', 'paragraph', 'text', 'is_comment') if field in row}})
    # Prefer the paragraph entry/bypass and contiguous context around actual
    # outline hits. Keep original line IDs; preview gaps remain explicit.
    entries, priority = defaultdict(list), defaultdict(list)
    for row in tools.evidence.values():
        if row.get('_artifact') != 'source.outline':
            continue
        key = (row.get('program'), row.get('source_file'))
        member = indexed.get(key, {})
        entry = row.get('first_line')
        anchors = [hit['line'] for hit in row.get('matches', []) if type(hit.get('line')) is int and hit['line'] in member]
        if type(entry) is int and entry in member and anchors:
            entries[key].append(range(entry, entry + 8))
            priority[key].extend(range(max(entry, line - 4), line + 13) for line in anchors[:2])
    for key, member in indexed.items():
        order = list(dict.fromkeys([line for span in entries[key] + priority[key] for line in span if line in member] + list(member)))
        groups[key[0]].extend(member[line] for line in order)
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
