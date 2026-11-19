"""Explain executed stages with bounded metadata, never model deliberation."""
import re

MODEL_PURPOSES = {
    'resolution': 'Identify the requested programs, scope and answer format before selecting evidence.',
    'decision': 'Use the returned evidence to choose the next read-only lookup or prepare a supported answer.',
    'tool_repair': 'Correct a failed tool request against its argument schema; a failed request is not evidence.',
    'answer_repair': 'Revise an answer rejected by validation using the recorded findings and evidence.',
    'evidence_recovery': 'Select additional evidence needed to address the previous validation findings.',
    'final_synthesis': 'Prepare an answer from the existing results; this stage is not opening new files.',
    'review': 'Independently compare the proposed answer, scope and citations against the retrieved evidence.',
    'partial_synthesis': 'Summarize supported findings and explicit gaps after exploration stopped or reached its budget.',
    'input_translation': 'Translate the question while preserving program names, identifiers and literal values.',
    'output_translation': 'Translate the supported answer while preserving code, identifiers, values and citations.',
    'output_translation_repair': 'Repair the answer language without changing its technical content.',
}


def model_context(phase, payload):
    """Only metadata from the payload actually being sent, not the entire corpus."""
    files, programs = set(), set()
    row_count = 0
    def add_row(row):
        nonlocal row_count
        if not isinstance(row, dict):
            return
        row_count += 1
        if row.get('program'):
            programs.add(str(row['program']))
        if row.get('source_file'):
            files.add((str(row.get('program') or ''), str(row['source_file'])))
    for observation in payload.get('observations', []):
        if not isinstance(observation, dict):
            continue
        args = observation.get('call', {}).get('args', {})
        if args.get('program'):
            programs.add(str(args['program']))
        programs.update(str(p) for p in args.get('programs', []) if isinstance(p, str))
        for row in observation.get('result', {}).get('rows', []):
            add_row(row)
    for key in ('retained_source_evidence', 'evidence'):
        block = payload.get(key, {})
        rows = block.get('rows', []) if isinstance(block, dict) else block if isinstance(block, list) else []
        for row in rows:
            add_row(row)
    return {'purpose': MODEL_PURPOSES.get(phase, 'Run the recorded model stage using the supplied request and evidence.'),
            'programs': sorted(programs)[:12],
            'context_files': [f'{program}: {name}' if program else name for program, name in sorted(files)[:12]],
            'context_rows': row_count}


def tool_context(tool, args, result=None):
    owner = args.get('program') or ', '.join(args.get('programs') or []) or 'the requested scope'
    if tool == 'source_outline':
        terms = ', '.join(args.get('terms') or args.get('verbs') or [])
        purpose = f'Locate recorded procedure paragraphs in {owner}' + (f' matching {terms}.' if terms else '.')
    elif tool in {'source', 'source_range'}:
        location = args.get('paragraph') or (f"lines {args['start']}–{args.get('end', args['start'])}" if 'start' in args else 'the requested source span')
        purpose = f'Read the implementation and surrounding conditions in {owner}, {location}.'
    elif tool == 'flow_edges':
        purpose = f'Check recorded control transfers and their conditions in {owner}.'
    elif tool == 'callers':
        purpose = 'Find recorded callers of the requested target, preserving call direction.'
    elif tool == 'callees':
        purpose = f'Find outgoing program calls made by {owner}.'
    elif tool == 'query':
        purpose = f"Read recorded {str(args.get('table', 'evidence')).replace('_', ' ')} for {owner}, applying the supplied filters."
    elif tool == 'search':
        purpose = f"Search {owner} using {args.get('mode', 'literal')} retrieval for the supplied terms."
    else:
        purpose = f"Run the read-only {tool.replace('_', ' ')} operation within its recorded scope."
    details = {'purpose': purpose}
    if result is not None:
        count = result.get('returned')
        if count is not None:
            details['outcome'] = f'{count} recorded matches returned.'
        if tool == 'source_outline':
            if count == 0:
                details['outcome'] = 'No matching recorded procedure paragraphs. This does not prove the requested behavior is absent.'
            else:
                details['matches'] = [dict(paragraph=row.get('name'), start=row.get('first_line'), end=row.get('last_line'))
                                      for row in result.get('rows', [])[:5]]
            details['next_hint'] = 'An outline is navigation only. Inspect matching source bodies or try a different recorded source search.'
        elif tool in {'source', 'source_range'}:
            if count == 0:
                details['outcome'] = 'No source lines returned for this location. Check the recorded member/paragraph; this is not proof of missing business logic.'
            elif result.get('requested_complete') is False:
                details['next_hint'] = 'The requested source was not returned completely. Check continuation spans or missing-line information.'
        elif count == 0:
            details['next_hint'] = 'No matches within this operation and filters; broader absence requires checking evidence coverage.'
    return details


def file_purpose(path, role):
    name = getattr(path, 'name', '')
    return {
        'program.source_lines.jsonl': 'Read indexed COBOL/copybook source text; this is not a fresh read of the original member or a new analysis run.',
        'architecture.call_parameters.json': 'Read recorded call targets, parameters and argument preparation.',
        'architecture.cics_operations.json': 'Read recorded CICS operations to add operation context to the selected source.',
        'controlflow.cfg.json': 'Read recorded paragraph transfers and conditions to add control-flow context.',
        'corpus.relationships.json': 'Read the published cross-program, COPY and JCL relationship graph.',
    }.get(name, f'Read the recorded {role}; this does not rerun the analyzer.')


def source_preview(rows):
    rows = [row for row in rows if row.get('source_file') and isinstance(row.get('text'), str)]
    meaningful = [row for row in rows if not row.get('is_comment') and
                  re.search(r'\b(?:IF|EVALUATE|MOVE|COMPUTE|ADD|SUBTRACT|MULTIPLY|DIVIDE|CALL|PERFORM|READ|WRITE|EXEC)\b',
                            str(row.get('normalized') or row.get('text', '')), re.I)]
    return [{key: row[key] for key in ('evidence_id', 'program', 'source_file', 'line',
                                      'line_start', 'paragraph', 'text') if key in row}
            for row in (meaningful or [row for row in rows if row.get('text', '').strip()])[:3]]
