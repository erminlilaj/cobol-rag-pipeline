"""Shared, data-driven contracts for the investigation model and executor."""

TABLE_INFO = {
    'variables': ('recorded variables', 'Variable declarations and use: name, controls_flow, origin, relationships, evidence.'),
    'calls': ('outgoing program calls', 'External program invocations: caller, target, call_type, paragraph, parameters, commarea, line_start. COPY is not a call_type.'),
    'cics': ('CICS operations', 'command, paragraph, resources, statement, line_start. Filter command for SEND, RECEIVE, SYNCPOINT, ABEND etc.'),
    'copybooks': ('COPY inclusions', 'Included copy members, not program invocations: copybook, line, section, division.'),
    'edges': ('control-flow edges', 'from, to, type, condition, evidence, line. Query destination to find incoming conditions.'),
    'paragraphs': ('source procedure paragraphs', 'Source paragraph names and source_file; excludes the implicit program entry node.'),
    'graph_nodes': ('control-flow graph nodes', 'Graph node names including implicit program entry; not a source paragraph count.'),
    'metrics': ('source metric records', 'physical_source_lines, source_procedure_paragraphs, mapa_paragraphs, graph_nodes; these measure different things.'),
    'literals': ('literal assignments', 'variable, value, paragraph, line_start, statement; includes initialization and forced values.'),
    'comments': ('classified comments', 'classification, text, paragraph, line. Disabled code: classification=commented_out_code. Do not search for the word comment.'),
    'summary': ('program overview records', 'Program identity, source header/purpose comments, recorded size, calls and CICS commands.'),
    'quality': ('quality-analysis records', 'Recorded commented-code and graph-reachability findings; absence of an incoming edge is not compiler proof of dead code.'),
    'copybook_review': ('copybook-review records', 'Copybook review candidates and analysis limitations; candidates are not proven unused.'),
    'files': ('source members per program', 'name, source_file, extension, recorded_lines. Includes main source and analyzed copybook source members.'),
    'source': ('physical source lines', 'source_file, line, text, paragraph, is_comment. Prefer the source tool for exact spans or paragraph bodies.'),
    'artifacts': ('analysis artifacts', 'name: generated analysis files, not COBOL source members.'),
}


def obj(properties, required=()):
    return dict(type='object', properties=properties, required=list(required), additionalProperties=False)


S = {'type': 'string'}
STRINGS = {'type': 'array', 'items': S}
FILTER = obj({'field': S, 'op': {'type': 'string', 'enum': ['eq', 'neq', 'in', 'contains']},
              'value': {'type': ['string', 'number', 'boolean', 'array', 'null']}}, ['field', 'op', 'value'])
PAGE = {'offset': {'type': 'integer', 'minimum': 0}, 'limit': {'type': 'integer', 'minimum': 0, 'maximum': 100}}
QUERY = {'programs': STRINGS, 'table': {'type': 'string', 'enum': list(TABLE_INFO)},
         'where': {'type': 'array', 'items': FILTER}, 'order_by': S, **PAGE}
TOOL_SCHEMAS = {
    'inventory': obj(PAGE),
    'files': obj({'programs': STRINGS, **PAGE}),
    'callers': obj({'target': S, 'programs': STRINGS, **PAGE}, ['target']),
    'quality': obj({'programs': STRINGS}, ['programs']),
    'describe': obj({'identifier': S, 'programs': STRINGS}, ['identifier']),
    'copybooks': obj({'programs': STRINGS, **PAGE}, ['programs']),
    'source_range': obj({'program': S, 'source_file': S, 'start': {'type': 'integer', 'minimum': 1},
                         'end': {'type': 'integer', 'minimum': 1}}, ['program', 'start', 'end']),
    'query': obj(QUERY, ['programs', 'table']),
    'select': obj({'result_id': S, 'basis': {'enum': ['collection', 'displayed']},
                   'where': QUERY['where'], 'order_by': S, **PAGE}, ['result_id', 'basis']),
    'compare': obj({'left_id': S, 'right_id': S, 'field': S,
                    'operation': {'enum': ['intersection', 'difference', 'union']}, **PAGE}, ['left_id', 'right_id', 'operation']),
    'inspect': obj({'evidence_id': S, 'program': S, 'artifact': S, 'field': S, **PAGE}),
    'source': obj({'program': S, 'source_file': S, 'paragraph': S,
                  'spans': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 2, 'maxItems': 2}}}, ['program']),
    'search': obj({'programs': STRINGS, 'text': S, 'mode': {'enum': ['literal', 'hybrid']}, 'limit': PAGE['limit']}, ['programs', 'text']),
}


def decision_schema():
    # A flat envelope avoids recursive unions in inference-server grammars.
    # Per-tool arguments are still strictly validated before execution.
    argument_fields = {key: value for spec in TOOL_SCHEMAS.values() for key, value in spec['properties'].items()}
    calls = {'type': 'array', 'maxItems': 3, 'items': obj({
        'tool': {'enum': list(TOOL_SCHEMAS)}, 'args': obj(argument_fields)}, ['tool', 'args'])}
    schema = obj({'action': {'enum': ['tools', 'final']}, 'calls': calls, 'answer': S,
                'mode': {'enum': ['technical', 'general', 'conversational', 'clarification']},
                'status': {'enum': ['complete', 'partial']}, 'requirements': STRINGS,
                'evidence_ids': STRINGS, 'result_id': {'type': ['string', 'null']},
                'collection_output': {'enum': ['list', 'count', 'summary']},
                'coverage': {'type': 'array', 'items': obj({'requirement': S, 'status': {'enum': ['answered', 'unavailable']}}, ['requirement', 'status'])}}, ['action'])
    schema['allOf'] = [
        {'if': {'properties': {'action': {'const': 'tools'}}}, 'then': {'required': ['calls']}},
        {'if': {'properties': {'action': {'const': 'final'}}}, 'then': {'required': ['answer', 'mode', 'status', 'evidence_ids']}},
    ]
    return schema


def tool_help():
    lines = ['Read-only tools (use the tool name and an args object):']
    for name, schema in TOOL_SCHEMAS.items():
        lines.append(name + '(' + ', '.join(schema['properties']) + ')')
    lines.extend(['inventory lists analyzed PROGRAMS, not files.',
                  'query reads a table for explicit programs; limit:0 counts, limit:N pages.',
                  'describe(identifier) resolves a program or entity and reads its evidence for explanations. Use it rather than answering from old citations.',
                  'copybooks(programs) reads actual COPY inclusions. Filename extensions do NOT identify all copybooks.',
                  'source_range(program,start,end) reads an inclusive line range; prefer this flat form for consecutive lines.',
                  'quality(programs) reads existing dead-code findings and copybook review evidence, including limitations. This analysis is available; do not claim there is no tool for it.',
                  'callers(target, programs optional) finds incoming calls TO target across the analyzed corpus; preserves caller, COMMAREA and parameters. query table=calls instead lists outgoing calls FROM owner programs.',
                  'select refines a saved result, basis collection or displayed. Preserve its filters.',
                  'source reads spans [[start,end]] or a paragraph body.',
                  'search mode literal finds source text; mode hybrid retrieves conceptual evidence.',
                  'Filters: {field,op:eq|neq|in|contains,value}. Multiple filters are AND.',
                  'Tables:'])
    lines.extend(f'{name}: {description}' for name, (_, description) in TABLE_INFO.items())
    return '\n'.join(lines)


def validate_tool(name, args):
    from jsonschema import validate, ValidationError
    if name not in TOOL_SCHEMAS:
        raise ValueError('Unknown tool; available: ' + ', '.join(TOOL_SCHEMAS))
    try:
        validate(args, TOOL_SCHEMAS[name])
    except ValidationError as error:
        raise ValueError(f'Invalid {name} arguments at {list(error.absolute_path)}: {error.message}') from error
