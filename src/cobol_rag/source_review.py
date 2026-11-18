"""Bounded review context from already retrieved source, never new evidence."""
import re


def review_context(tools, entities=()):
    focus = [{'program': e.get('program'), 'name': e['name'], 'kind': e.get('entity_type')}
             for e in entities if e.get('entity_type') not in {'program', 'programs'}][:12]
    names = {e['name'].upper() for e in focus}
    members = {}
    for row in tools.evidence.values():
        if (row.get('source_file') and type(row.get('line')) is int and isinstance(row.get('text'), str)
                and not row.get('is_comment')):
            key = (row.get('program'), row['source_file'])
            members.setdefault(key, {})[row['line']] = row
    branches, operations = [], []
    for (program, member), rows in members.items():
        for line, row in sorted(rows.items()):
            text = str(row.get('normalized') or row['text']).strip()
            if re.match(r'^(?:OPEN|READ|WRITE|CLOSE|EXEC\s+CICS)\b', text, re.I):
                operations.append({'program': program, 'source_file': member, 'line': line,
                                   'text': text[:240]})
            if not re.match(r'^IF\s', text, re.I):
                continue
            span, complete = [], False
            for number in range(line, line + 24):
                if number not in rows:
                    break
                source = rows[number]['text']
                span.append({'line': number, 'text': source})
                unquoted = re.sub(r"'(?:[^']|'')*'|\"[^\"]*\"", '', source)
                if unquoted.rstrip().endswith('.'):
                    complete = True
                    break
            branches.append({'program': program, 'source_file': member, 'start': line,
                             'contains_focal_name': any(re.search(r'(?<![A-Z0-9-])' + re.escape(name) +
                                                                r'(?![A-Z0-9-])', text.upper()) for name in names),
                             'recorded_through_period': complete, 'source': span})
    branches.sort(key=lambda row: (not row['contains_focal_name'], row['start']))
    # Source remains bounded; completeness is only that of this recorded window.
    selected = []
    owners = {}
    for row in branches:
        owners.setdefault(row['program'], []).append(row)
    interleaved = [rows[index] for index in range(max((len(rows) for rows in owners.values()), default=0))
                   for rows in owners.values() if index < len(rows)]
    for row in interleaved[:4]:
        row = {**row, 'source': row['source'][:16]}
        row['recorded_through_period'] = row['recorded_through_period'] and len(row['source']) < 16
        selected.append(row)
    accesses = []
    for row in tools.evidence.values():
        recipe = row.get('recipe', {})
        if recipe.get('tool') == 'query' and recipe.get('args', {}).get('table') == 'variable_access':
            accesses.append({'query': recipe['args'], 'matches': row.get('total_matches'),
                             'storage_scope': 'Exact queried identifiers only; excludes unqueried child fields, aliases and callee side effects.'})
    return {'focal_entities': focus, 'condition_windows': selected,
            'file_and_cics_operations': operations[:12], 'access_scopes': accesses[-4:],
            'limitation': 'Already retrieved static source only. A source window is not a runtime proof; verify the complete condition, true/false scope and exact destination.'}
