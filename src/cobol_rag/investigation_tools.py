"""Read-only evidence primitives for the opt-in investigation workflow."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from cobol_rag import final_scripts_answers as artifacts
from cobol_rag.investigation_protocol import TABLE_INFO, validate_tool


class ToolError(ValueError):
    """An actionable tool error, not permission to substitute other evidence."""


def source_semantics(records):
    """Preserve physical text, but do not treat trailing headings as paragraph behavior."""
    rows = deepcopy(records)
    for row in rows:
        text = str(row.get('normalized') or row.get('text') or '').strip()
        if row.get('is_comment'):
            row['source_role'] = 'comment_not_executable'
        elif re.match(r'^(?:SKIP[123]?|EJECT)\b', text, re.I):
            row['source_role'] = 'listing_directive_not_executable'
        elif re.match(r'^(?:COPY\b|EXEC\s+SQL\s+INCLUDE\b)', text, re.I):
            row['source_role'] = 'inclusion_directive_expand_for_runtime_behavior'
        else:
            row['source_role'] = 'source_statement'
    # Source maps commonly carry the previous paragraph across the heading of
    # the next one. Keep those physical lines available, without asserting ownership.
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        if not row.get('paragraph') or not (row.get('is_comment') or row.get('is_blank')):
            continue
        following = rows[index + 1] if index + 1 < len(rows) else {}
        if (following.get('source_file'), following.get('paragraph')) != (row.get('source_file'), row.get('paragraph')):
            row['recorded_paragraph'] = row['paragraph']
            row['paragraph'] = None
            row['source_role'] = 'boundary_comment_or_blank_not_paragraph_behavior'
    return rows


class EntityTypeMismatch(ToolError):
    def __init__(self, message, programs, matches):
        super().__init__(message)
        self.programs, self.matches = programs, matches


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def field(row, path, default=None):
    value = row
    for name in path.split('.'):
        if not isinstance(value, dict) or name not in value:
            return default
        value = value[name]
    return value


def match(row, predicate):
    key = predicate.get('field', '')
    op = predicate.get('op', 'eq')
    wanted = predicate.get('value')
    if not key or op not in {'eq', 'neq', 'in', 'contains'}:
        raise ToolError('Filter needs field, op=eq|neq|in|contains, and value.')
    missing = object()
    actual = field(row, key, missing)
    if actual is missing:
        raise ToolError(f'Field {key} is unavailable; inspect the schema first.')
    normalize = lambda x: str(x).casefold()
    if op == 'contains':
        return normalize(wanted) in normalize(actual)
    if op == 'in':
        if not isinstance(wanted, list):
            raise ToolError('An in filter requires an array.')
        return normalize(actual) in {normalize(v) for v in wanted}
    equal = normalize(actual) == normalize(wanted)
    return equal if op == 'eq' else not equal


TABLES = {
    'variables': ('dataflow.used_variables.json', 'variables'),
    'variable_access': ('dataflow.used_variables.json', 'variables'),
    'calls': ('architecture.call_parameters.json', 'calls'),
    'parameter_writes': ('architecture.call_parameters.json', 'calls'),
    'cics': ('architecture.cics_operations.json', 'content.operations'),
    'copybooks': ('architecture.copybooks.json', 'content.inclusions'),
    'edges': ('controlflow.cfg.json', 'edges'),
    'graph_nodes': ('controlflow.cfg.json', 'nodes'),
    'literals': ('dataflow.literal_assignments.json', 'assignments'),
    'comments': ('program.comments.json', 'comments'),
    'summary': ('program.summary.json', ''),
    'quality': ('quality.dead_code.json', ''),
    'copybook_review': ('architecture.unused_copybooks.json', ''),
}

# Measurements keep their meaning and provenance at the evidence boundary.
# This vocabulary validates answers; it never routes a user's question.
METRIC_DEFINITIONS = {
    'mapa_loc': ('MAPA LOC', 'program.summary.json'),
    'mapa_paragraphs': ('MAPA paragraphs', 'program.summary.json'),
    'source_procedure_paragraphs': ('source procedure paragraphs', 'program.comments.json'),
    'physical_source_lines': ('physical source lines', 'program.comments.json'),
    'graph_nodes': ('control-flow graph nodes', 'controlflow.cfg.json'),
}


def compact(value, depth=0):
    """Explicit previews; full evidence stays available through inspect."""
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + ' [PREVIEW TRUNCATED: inspect this field]'
    if isinstance(value, list):
        cap = 5 if depth else 12
        return [compact(v, depth + 1) for v in value[:cap]] + ([{'more_items': len(value) - cap}] if len(value) > cap else [])
    if isinstance(value, dict):
        return {k: compact(v, depth + 1) for k, v in value.items()}
    return value


class EvidenceTools:
    """Bounded session handles contain replay recipes, not whole inventories."""
    def __init__(self, config, memory=None):
        self.config = config
        self.programs = tuple(artifacts.analyzed_programs())
        self.memory = deepcopy(memory or {'collections': {}, 'last_result_id': None})
        self.memory.setdefault('collections', {})
        self.evidence = {}
        self._ids = {}
        self._cache = {}
        self.group_contexts = []

    def mentioned_entities(self, text):
        """Reuse the existing corpus catalogue, without inferring question intent."""
        from cobol_rag.scope import _catalogue
        root = artifacts.find_final_scripts_root()
        _, entities = _catalogue(str(root) if root else '')
        tokens = set(re.findall(r'[A-Za-z0-9_$.-]+', text.upper()))
        tokens |= {token.rstrip('.') for token in tokens}
        matches = [{'name': p, 'program': p, 'entity_type': 'program'}
                   for p in self.programs if p in tokens]
        matches.extend({'name': e.value, 'program': e.program, 'entity_type': e.entity_type}
                       for e in entities if e.value in tokens)
        return matches

    def root(self, program):
        program = str(program).upper()
        if program not in self.programs:
            raise ToolError(f'Unknown analyzed program {program}; available: {self.programs}.')
        base = artifacts.find_final_scripts_root()
        root = artifacts.find_program_artifact_root(base, program) if base else None
        if root is None:
            raise ToolError(f'Analysis unavailable for {program}.')
        return root

    def path(self, program, name):
        root = self.root(program).resolve()
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ToolError('Only relative artifacts within the selected program may be read.')
        path = artifacts._artifact_path(root, name).resolve()
        if not path.is_relative_to(root):
            raise ToolError('Artifact escapes the selected program.')
        if not path.is_file():
            raise ToolError(f'Analysis gap: {program}/{name} is unavailable.')
        if path.stat().st_size > 20_000_000:
            raise ToolError('Artifact exceeds the tool size limit; inspect a smaller artifact.')
        return path

    def load(self, program, name):
        path = self.path(program, name)
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if key not in self._cache:
            self._cache[key] = json.loads(path.read_text(encoding='utf-8'))
        return deepcopy(self._cache[key])

    def rows(self, program, table):
        program = str(program).upper()
        root = self.root(program)
        if table in {'files', 'source', 'paragraphs'}:
            self.path(program, 'program.source_lines.jsonl')
            raw = source_semantics(artifacts._read_source_lines(root, program))
            if not raw:
                raise ToolError('Analysis gap: physical source map unavailable.')
            artifact = 'program.source_lines.jsonl'
            if table == 'paragraphs':
                names = sorted({(r['source_file'], r.get('paragraph')) for r in raw
                                if r.get('paragraph') and r.get('paragraph') != program
                                and r.get('division') == 'PROCEDURE DIVISION'
                                and r['source_file'].upper() == program + '.CBL'})
                raw = [{'name': name, 'source_file': source} for source, name in names]
            if table == 'files':
                names = sorted({r['source_file'] for r in raw})
                raw = [{'name': n, 'source_file': n, 'extension': Path(n).suffix,
                        'recorded_lines': sum(r['source_file'] == n for r in raw)} for n in names]
        elif table in {'summary', 'metrics'}:
            artifact = 'program.summary.json'
            summary = self.load(program, artifact)
            comments = self.load(program, 'program.comments.json')
            cfg = self.load(program, 'controlflow.cfg.json')
            metrics = {'mapa_loc': summary.get('meta', {}).get('loc'),
                       'physical_source_lines': comments.get('metrics', {}).get('total_lines'),
                       'source_procedure_paragraphs': comments.get('metrics', {}).get('total_procedure_paragraphs'),
                       'mapa_paragraphs': summary.get('meta', {}).get('paragraphs'),
                       'graph_nodes': len(cfg.get('nodes', []))}
            if table == 'metrics':
                raw = [metrics]
            else:
                calls = self.load(program, 'architecture.call_parameters.json')
                raw = [{'identity': summary.get('content'),
                        'purpose_comments': [r.get('normalized_text', r.get('text')) for r in comments.get('comments', [])
                                             if r.get('line', 9999) < 30 and r.get('indexable')],
                        'metrics': metrics, 'outgoing_calls': [r.get('target') for r in calls.get('calls', [])],
                        'limitations': 'Purpose comments describe intent; graph nodes and source paragraphs are different measurements.'}]
            raw[0]['metric_facts'] = [dict(field=key, value=metrics[key], unit=unit, source_artifact=source)
                for key, (unit, source) in METRIC_DEFINITIONS.items() if metrics.get(key) is not None]
            raw[0]['source_artifacts'] = ['program.summary.json', 'program.comments.json', 'controlflow.cfg.json']
            if table == 'summary':
                raw[0]['source_artifacts'].append('architecture.call_parameters.json')
        elif table == 'artifacts':
            artifact = 'artifact_inventory'
            raw = [{'name': str(p.relative_to(root))} for p in sorted(root.rglob('*.json'))
                   if p.resolve().is_relative_to(root.resolve())]
        elif table in TABLES:
            artifact, key = TABLES[table]
            payload = self.load(program, artifact)
            raw = field(payload, key) if key else [payload]
            if raw is None and isinstance(payload, dict):
                raw = payload.get(key.split('.')[-1])
            if not isinstance(raw, list):
                raise ToolError(f'Analysis schema gap: {artifact} has no row collection {key}; inspect it.')
            if table == 'parameter_writes':
                raw = [dict(site, caller=call.get('caller', program), target=call.get('target'),
                            parameter=detail.get('parameter'), variable=variable.get('variable'),
                            name=variable.get('variable'), call_line=call.get('line_start'))
                       for call in raw for detail in call.get('parameter_details', [])
                       for variable in detail.get('variables', [])
                       for site in variable.get('writes_before_call', [])]
            elif table == 'copybooks':
                classifications = payload.get('content', {}).get('classified', {})
                raw = [dict(row, categories=[category for category, members in classifications.items()
                            if row.get('copybook') in members],
                            classification_note=payload.get('content', {}).get('classification_note', ''))
                       for row in raw]
                # Join recorded review facts by identity, not by question wording.
                # Missing analysis stays unknown; inclusion alone proves no status.
                try:
                    review = self.load(program, 'architecture.unused_copybooks.json').get('content', {})
                    if isinstance(review.get('needs_review_copybooks'), list) and isinstance(review.get('unused_copybooks_proven'), list):
                        candidates = {str(n).upper() for n in review['needs_review_copybooks']}
                        proven = {str(n).upper() for n in review['unused_copybooks_proven']}
                        for row in raw:
                            name = str(row.get('copybook', '')).upper()
                            row.update(needs_review=name in candidates, proven_unused=name in proven,
                                review_source='architecture.unused_copybooks.json',
                                review_proof_level=review.get('proof_level'),
                                review_limitations=review.get('limitations', []))
                except (ToolError, OSError):
                    pass
            elif table == 'variable_access':
                raw = [dict(site, name=variable.get('variable'), variable=variable.get('variable'),
                            access_kind=kind.removesuffix('_sites'))
                       for variable in raw for kind, sites in variable.get('evidence', {}).items()
                       if isinstance(sites, list) for site in sites if isinstance(site, dict)]
            elif table == 'copybook_review':
                content = payload.get('content', {})
                raw = [{key: content[key] for key in ('copybooks_total', 'all_copybooks',
                    'needs_review_count', 'needs_review_copybooks', 'unused_copybooks_proven',
                    'proof_level', 'limitations') if key in content}]
            elif table == 'quality':
                raw = [payload.get('content', payload)]
            elif table == 'literals':
                raw = [dict(row, variable=row.get('target_variable', row.get('variable')),
                            value=row.get('literal', row.get('value')),
                            line_start=row.get('line', row.get('line_start'))) for row in raw]
        else:
            raise ToolError('Unknown table; available: ' + ', '.join([*TABLES, 'files', 'source', 'artifacts']))
        result = []
        for item in raw:
            row = dict(item) if isinstance(item, dict) else {'name': str(item)}
            if table == 'variable_access' and isinstance(row.get('line_start'), int) and row['line_start'] < 1:
                row['recorded_line_start'] = row['line_start']
                row['line_start'] = None
                row['address_status'] = 'No physical source address recorded'
                row['access_kind'] = 'unlocated_' + row['access_kind']
            row.setdefault('name', row.get('variable') or row.get('target') or row.get('copybook') or row.get('command') or table)
            row.update(program=program, _artifact=artifact)
            if table == 'source' and row.get('division') == 'DATA DIVISION':
                declaration = re.match(r'^\s*\**\s*(0[1-9]|[1-4][0-9]|66|77|78|88)\s+([A-Z0-9][A-Z0-9-]*)\b',
                                       str(row.get('normalized', row.get('text', ''))), re.I)
                if declaration:
                    row['declaration'] = {'level': int(declaration.group(1)),
                        'name': declaration.group(2).upper(), 'active': not row.get('is_comment', False),
                        'basis': 'source declaration syntax; includes group items without PIC'}
            if table == 'edges':
                # CFG CALL is a graph edge category, not the COBOL CALL verb.
                statement = str(row.get('evidence', '')).strip()
                verb = re.match(r'(PERFORM|GO\s+TO|CALL)\b', statement, re.I)
                row['source_operation'] = ' '.join(verb.group(1).upper().split()) if verb else None
                row['execution_limitations'] = 'Static edge, not proof of a feasible runtime path; fallthrough after a terminating operation may be infeasible.'
            row['_row_id'] = digest(row)
            result.append(row)
        return result

    def evaluate(self, recipe, depth=0):
        if depth > 6:
            raise ToolError('Result reference nesting exceeded; issue a fresh query.')
        tool, args = recipe['tool'], recipe['args']
        if tool == 'inventory':
            rows = []
            for p in self.programs:
                row = {'name': p, 'program': p, '_artifact': 'corpus.registry.json'}
                try:
                    members = self.rows(p, 'files')
                    row['source_members'] = [r['source_file'] for r in members]
                    row['source_member_count'] = len(members)
                    row['source_members_evidence'] = 'program.source_lines.jsonl'
                except (ToolError, OSError) as error:
                    row['source_members_unavailable'] = str(error)
                row['_row_id'] = digest(row)
                rows.append(row)
        elif tool == 'query':
            programs = args.get('programs') or []
            if not isinstance(programs, list) or not programs:
                raise ToolError('query needs programs, for example ["PDCBVC"].')
            rows = [r for p in dict.fromkeys(programs) for r in self.rows(p, args['table'])]
        elif tool == 'select':
            rows, _ = self.evaluate(args['parent'], depth + 1)
            if args['basis'] == 'displayed':
                by_id = {r['_row_id']: r for r in rows}
                rows = [by_id[i] for i in args['displayed_ids'] if i in by_id]
        elif tool == 'compare':
            left, _ = self.evaluate(args['left'], depth + 1)
            right, _ = self.evaluate(args['right'], depth + 1)
            key = args.get('field', 'name')
            if any(field(r, key) is None for r in left + right):
                raise ToolError('Comparison field unavailable.')
            a = {str(field(r, key)).upper() for r in left}
            b = {str(field(r, key)).upper() for r in right}
            op = args.get('operation')
            values = a & b if op == 'intersection' else a - b if op == 'difference' else a | b if op == 'union' else None
            if values is None:
                raise ToolError('Comparison requires intersection, difference or union.')
            rows = [{'name': v, 'members': [{k: r[k] for k in
                     ('name', 'program', 'caller', 'target', 'call_type', 'paragraph', 'line_start', '_artifact', '_row_id') if k in r}
                     for r in left + right if str(field(r, key)).upper() == v],
                     '_artifact': 'derived_set_operation', '_row_id': digest((recipe, v))} for v in sorted(values)]
        else:
            raise ToolError('Invalid replay operation.')
        category_values = {key: {str(field(r, key)).casefold() for r in rows if field(r, key) is not None}
                           for key in ('call_type', 'command', 'classification')}
        schema_rows = rows
        for predicate in args.get('where', []):
            key = predicate.get('field', '')
            missing = object()
            actual_values = [field(r, key, missing) for r in schema_rows]
            present = [v for v in actual_values if v is not missing]
            wanted = predicate.get('value')
            values = wanted if isinstance(wanted, list) else [wanted]
            if present and all(isinstance(v, bool) for v in present) and any(str(v).lower() not in {'true', 'false'} for v in values):
                raise ToolError(f'Field {key} is boolean; use true or false as the value, not an operator or field name.')
            if predicate.get('field') == 'origin' and predicate.get('op') in {'eq', 'in'}:
                values = predicate.get('value')
                values = values if isinstance(values, list) else [values]
                origins = {str(r.get('origin', '')).upper() for r in rows}
                if any(str(v).upper() in self.programs and str(v).upper() not in origins for v in values):
                    error = ToolError('Program scope is already selected by programs. origin is a declaration origin, not the owning program; remove this ownership filter or use a recorded origin.')
                    error.suggested_where = [p for p in args.get('where', []) if p != predicate]
                    raise error
            if schema_rows and not present:
                hint = (' Use quality(programs) for reachability/unused findings, not paragraph inventory filters.'
                        if args.get('table') == 'paragraphs' else '')
                raise ToolError(f'Field {key} is unavailable; inspect the table schema before filtering.' + hint)
            if tool == 'query' and args.get('table') in {'variables', 'copybooks', 'paragraphs'} and predicate.get('field') == 'name':
                values = predicate.get('value')
                values = values if isinstance(values, list) else [values]
                names = {str(r.get('name', '')).upper() for r in rows}
                if any(str(v).upper() in self.programs and str(v).upper() not in names for v in values):
                    raise ToolError('Entity type mismatch: this filter names a program, not an entity in this table. Program scope is already set by programs; do not use it as a row-name filter.')
                if predicate.get('op', 'eq') in {'eq', 'in'}:
                    missing = {str(v).upper() for v in values} - names
                    alternatives = {}
                    for other in ('variables', 'paragraphs', 'copybooks', 'calls'):
                        if not missing or other == args['table']:
                            continue
                        try:
                            known = {str(r['name']).upper() for p in programs for r in self.rows(p, other)}
                        except (ToolError, OSError, KeyError):
                            continue
                        for name in missing & known:
                            alternatives.setdefault(name, []).append(other)
                    if alternatives:
                        raise EntityTypeMismatch(f'Entity type mismatch: {alternatives} are recorded in other tables, not {args["table"]}. Reconsider the entity type. No query has been silently redirected.', programs, alternatives)
            # Categorical values are constrained by the artifact vocabulary.
            # An invalid category is not a legitimate empty result.
            if predicate.get('field') in category_values and category_values[predicate['field']]:
                allowed = category_values[predicate['field']]
                requested = predicate.get('value')
                requested = requested if isinstance(requested, list) else [requested]
                if predicate.get('op', 'eq') in {'eq', 'in'} and any(str(v).casefold() not in allowed for v in requested):
                    hint = ''
                    if args.get('table') == 'cics' and predicate['field'] == 'command':
                        paragraphs = {str(r.get('name', '')).casefold() for p in programs for r in self.rows(p, 'paragraphs')}
                        matches = [str(v) for v in requested if str(v).casefold() in paragraphs]
                        if matches:
                            hint = f' These are paragraph names: {matches}. Use flow_edges for entry/exit relationships or source(program, paragraph) for the body.'
                    raise ToolError(f'Invalid {predicate["field"]} category; recorded values: {sorted(allowed)}. Choose the correct table; COPY inclusions use copybooks.' + hint)
            rows = [r for r in rows if match(r, predicate)]
        order = args.get('order_by', 'name' if tool != 'select' else None)
        if order:
            descending = order.startswith('-')
            order = order[1:] if descending else order
            if rows and any(field(r, order) is None for r in rows):
                raise ToolError(f'Ordering field {order} unavailable.')
            rows.sort(key=lambda r: (field(r, order), r.get('program', ''), r['_row_id']), reverse=descending)
        return rows, digest(rows)

    def register(self, row):
        key = digest(row)
        if key not in self._ids:
            eid = f'E{len(self.evidence) + 1}'
            self._ids[key] = eid
            self.evidence[eid] = deepcopy(row)
        return self._ids[key]

    def call_relation(self, recipe):
        """Describe the executed relation, including a zero-row result."""
        args = recipe['args']
        if recipe['tool'] == 'compare':
            inputs = [self.call_relation(args[side]) for side in ('left', 'right')]
            if all(inputs):
                return {'operation': args['operation'], 'inputs': inputs,
                        'absence_scope': 'Derived set over the recorded input relationships only.'}
        if recipe['tool'] == 'select':
            relation = self.call_relation(args['parent'])
            if relation:
                relation = {**relation, 'refinement_filters': args.get('where', [])}
            return relation
        if recipe['tool'] != 'query' or args.get('table') != 'calls':
            return None
        target_filters = [p for p in args.get('where', []) if p.get('field') == 'target']
        return {'edge': 'caller -> target', 'caller_scope': args['programs'],
                'target_filters': target_filters, 'filters': args.get('where', []),
                'direction': 'incoming_to_target' if target_filters else 'outgoing_from_callers',
                'absence_scope': 'Only this caller scope and these filters were checked; the reverse direction was not checked.'}

    def paragraph_operations(self, program, paragraph):
        """Join analyzed operations to their host paragraph, including COPY origin."""
        try:
            rows = self.rows(program, 'cics')
        except ToolError:
            return []
        selected = [r for r in rows if str(r.get('paragraph', '')).casefold() == paragraph.casefold()]
        return [{'evidence_id': self.register(row), **{key: row[key] for key in
                 ('program', 'paragraph', 'command', 'source_file', 'line_start', 'line_end', 'included_at_line', 'statement') if key in row},
                 'execution_semantics': {
                     'ABEND': 'Abnormal termination; a static fallthrough edge does not establish normal continuation after this operation.',
                     'LINK': 'Invoke the named program with the specified COMMAREA; LENGTH supplies its byte length, not a subsequent computation or check.',
                 }.get(str(row.get('command', '')).upper(), 'Preserve the full operation and all its operands.')}
                for row in selected]

    def paragraph_flow_context(self, program, paragraph):
        """Attach both edge roles without confusing entry conditions with body."""
        try:
            rows = self.rows(program, 'edges')
        except (ToolError, OSError) as error:
            return {'unavailable': str(error)}
        result = {'evidence_scope': 'Recorded static control-flow edges, not runtime reachability.',
                  'entry_condition_basis': 'Conditions on incoming edges whose to equals this paragraph. '
                  'Completeness covers the recorded edge inventory only, not every feasible execution path.'}
        for direction, key in (('incoming', 'to'), ('outgoing', 'from')):
            selected = [r for r in rows if str(r.get(key, '')).casefold() == paragraph.casefold()]
            result[direction] = {'total': len(selected), 'complete': len(selected) <= 20,
                'rows': [{'evidence_id': self.register(r), **r} for r in selected[:20]]}
        return result

    def data_dependencies(self, args):
        """Bounded may-dependency paths from co-recorded reads and writes.

        No execution feasibility or reaching-definition claim is inferred.
        Matching location AND statement prevents joins across unrelated sites.
        """
        paths = []
        source, target = args['source'].upper(), args['target'].upper()
        depth_limit = args.get('max_depth', 4)
        for program in args['programs']:
            definitions = {str(r.get('name', '')).upper(): r for r in self.rows(program, 'variables')}
            sites = {}
            for row in self.rows(program, 'variable_access'):
                if not row.get('line_start') or not row.get('statement'):
                    continue
                key = (row.get('source_file'), row.get('paragraph'), row['line_start'], row['statement'])
                site = sites.setdefault(key, {'read': set(), 'write': set(), 'row': row})
                kind = row.get('access_kind')
                if kind in {'read', 'read_write'}:
                    site['read'].add(str(row['variable']).upper())
                if kind in {'write', 'read_write'}:
                    site['write'].add(str(row['variable']).upper())
            adjacency = {}
            for site in sites.values():
                for reader in site['read']:
                    for writer in site['write'] - {reader}:
                        r = site['row']
                        edge = {k: r[k] for k in ('program', 'source_file', 'paragraph', 'line_start', 'statement') if k in r}
                        adjacency.setdefault(reader, []).append({**edge, 'from': reader, 'to': writer})
            queue = [(source, [], {source})]
            visited_states = 0
            while queue and len(paths) < 20 and visited_states < 1000:
                node, path, seen = queue.pop(0)
                visited_states += 1
                if len(path) >= depth_limit:
                    continue
                for edge in adjacency.get(node, []):
                    if edge['to'] in seen:
                        continue
                    next_path = path + [edge]
                    if edge['to'] == target:
                        names = list(dict.fromkeys([source] + [e['to'] for e in next_path]))
                        paths.append({'program': program, 'source': source, 'target': target, 'steps': next_path,
                                      'variables': [{'name': name, 'origin': definitions.get(name, {}).get('origin'),
                                          'declarations': definitions.get(name, {}).get('relationships', {}).get('declarations', [])}
                                          for name in names],
                                      '_artifact': 'derived_data_dependency'})
                    else:
                        queue.append((edge['to'], next_path, seen | {edge['to']}))
        return {'rows': [{'evidence_id': self.register(p), **p} for p in paths[:20]],
                'returned': min(len(paths), 20), 'max_depth': depth_limit,
                'limitation': 'Static may-dependencies from matching read/write sites, not proof of runtime feasibility or reaching definitions. At most 20 paths and 1000 search states per program; no path is not proof of no dependency. Inspect source conditions and declaration formats to explain behavior.'}

    def saved(self, rid):
        saved = self.memory['collections'].get(rid)
        if saved is None:
            raise ToolError('Unknown result set; use the conversation result handles.')
        _, fingerprint = self.evaluate(saved['recipe'])
        if fingerprint != saved['fingerprint']:
            raise ToolError('Corpus changed since this result; issue a fresh query.')
        return saved

    def execute(self, tool, args):
        if not isinstance(args, dict):
            raise ToolError('Tool arguments must be an object.')
        args = deepcopy(args)
        try:
            validate_tool(tool, args)
        except ValueError as error:
            raise ToolError(str(error)) from error
        if tool == 'data_dependencies':
            return self.data_dependencies(args)
        if tool == 'files':
            return self.execute('query', {**args, 'programs': args.get('programs') or list(self.programs), 'table': 'files'})
        if tool == 'flow_edges':
            direction, paragraph = args.pop('direction'), args.pop('paragraph')
            return self.execute('query', {**args, 'table': 'edges', 'where': [
                {'field': 'to' if direction == 'incoming' else 'from', 'op': 'eq', 'value': paragraph}]})
        if tool == 'variable_access':
            variables = args.pop('variables')
            if any(not name or '*' in name or '?' in name for name in variables):
                raise ToolError('variable_access requires literal variable identifiers, not wildcards. '
                                'For a variable inventory use query table=variables; controls_flow is a boolean property. '
                                'For all access sites use query table=variable_access without an identifier filter.')
            kind = args.pop('access_kind', None)
            where = [{'field': 'variable', 'op': 'in', 'value': variables}]
            if kind:
                where.append({'field': 'access_kind', 'op': 'eq', 'value': kind})
            return self.execute('query', {**args, 'table': 'variable_access', 'where': where})
        if tool == 'copybooks':
            return self.execute('query', {**args, 'table': 'copybooks'})
        if tool == 'callees':
            target = args.pop('target', None)
            return self.execute('query', {**args, 'table': 'calls',
                **({'where': [{'field': 'target', 'op': 'in' if isinstance(target, list) else 'eq', 'value': target}]} if target else {})})
        if tool == 'source_range':
            start, end = args.pop('start'), args.pop('end')
            return self.execute('source', {**args, 'spans': [[start, end]]})
        if tool == 'group_context':
            program, group = args['program'].upper(), args['group'].upper()
            source_prefix = str(args.get('source_prefix') or '').upper().rstrip('-')
            filename = program + '.CBL'
            source_rows = [r for r in self.rows(program, 'source')
                           if str(r.get('source_file', '')).upper() == filename]
            declarations = [(index, row) for index, row in enumerate(source_rows)
                            if row.get('division') == 'DATA DIVISION'
                            and (row.get('declaration') or {}).get('name') == group
                            and (row.get('declaration') or {}).get('active')]
            if len(declarations) != 1:
                raise ToolError(f'Expected one active destination data-group declaration for {group}; found {len(declarations)}. '
                                'For fields of an interface shown in a map row, group is the map-row data group '
                                'and source_prefix is the interface name. Correct those roles before retrying.')
            start, parent = declarations[0]
            level = parent['declaration']['level']
            layout = [parent]
            for row in source_rows[start + 1:]:
                if row.get('division') != 'DATA DIVISION':
                    break
                declaration = row.get('declaration') or {}
                if declaration.get('active') and declaration.get('level', 99) <= level:
                    break
                if declaration.get('active') and declaration.get('level', 0) > level:
                    layout.append(row)
            members = {row['declaration']['name'] for row in layout}
            group_pattern = re.compile(r'(?<![A-Z0-9-])' + re.escape(group) + r'(?![A-Z0-9-])')
            procedures = [row for row in source_rows if row.get('division') == 'PROCEDURE DIVISION'
                          and row.get('paragraph') and not row.get('is_comment')]
            hosts = {row['paragraph'] for row in procedures
                     if group_pattern.search(str(row.get('normalized', row.get('text', ''))).upper())}
            names = [re.compile(r'(?<![A-Z0-9-])' + re.escape(name) + r'(?![A-Z0-9-])')
                     for name in members]
            relevant = [row for row in procedures if row['paragraph'] in hosts and
                        (any(pattern.search(str(row.get('normalized', row.get('text', ''))).upper())
                             for pattern in names) or
                         source_prefix + '-' in str(row.get('normalized', row.get('text', ''))).upper()
                         if source_prefix else any(pattern.search(
                             str(row.get('normalized', row.get('text', ''))).upper()) for pattern in names))]
            selected = [*layout, *relevant]
            source_fields = sorted({name for row in relevant
                for name in re.findall(r'\b' + re.escape(source_prefix) + r'-[A-Z0-9-]+\b',
                    str(row.get('normalized', row.get('text', ''))).upper())}) if source_prefix else []
            self.group_contexts.append({'program': program, 'group': group,
                                        'source_prefix': source_prefix,
                                        'candidate_source_fields': source_fields})
            output_moves = [row for row in relevant if re.search(
                r'\bMOVE\s+' + re.escape(group) + r'\s+TO\b',
                str(row.get('normalized', row.get('text', ''))).upper())]
            return {'rows': [{'evidence_id': self.register(row),
                              **{key: row[key] for key in ('program', 'source_file', 'line',
                                  'paragraph', 'text', 'declaration') if key in row}}
                             for row in selected[:80]],
                    'returned': min(len(selected), 80), 'total_matches': len(selected),
                    'group': group, 'member_names': sorted(members - {group, 'FILLER'}),
                    'preparation_paragraphs': sorted(hosts), 'source_prefix': source_prefix or None,
                    'candidate_source_fields': source_fields,
                    'output_transfers': [{'line': row['line'], 'paragraph': row['paragraph'],
                                          'statement': row['text'].strip()} for row in output_moves],
                    'complete': len(selected) <= 80,
                    'limitation': 'Relevant source lines, not a proven complete value-flow graph. '
                                  'Inspect surrounding statements for intermediate values and conditions.'}
        if tool == 'describe':
            identifier = args['identifier'].upper()
            if identifier in self.programs:
                result = self.execute('query', {'programs': [identifier], 'table': 'summary'})
                if args.get('depth') == 'detailed':
                    result['detail_coverage'] = {}
                    for table in ('calls', 'edges', 'cics'):
                        try:
                            records = self.rows(identifier, table)
                            result['detail_coverage'][table] = {'total': len(records), 'returned': min(8, len(records))}
                            for record in records[:8]:
                                row = {k: record[k] for k in ('program', '_artifact', 'name', 'caller', 'target',
                                    'paragraph', 'line_start', 'line', 'statement', 'parameters', 'commarea', 'length', 'call_type',
                                    'from', 'to', 'type', 'condition', 'evidence', 'command') if k in record}
                                if table == 'calls':
                                    row['relation'] = self.call_relation({'tool': 'query', 'args': {'programs': [identifier], 'table': 'calls'}})
                                    row['call_inventory_counts'] = {kind: sum(r.get('call_type') == kind for r in records)
                                        for kind in {r.get('call_type') for r in records} if kind}
                                result['rows'].append({'evidence_id': self.register(row), **row})
                        except (ToolError, OSError) as exc:
                            result['detail_coverage'][table] = {'unavailable': str(exc)}
                    result['returned'] = len(result['rows'])
                    result['detail_limitations'] = 'Behavioral samples, not exhaustive paths or full implementation proof. Query specific aspects for more detail.'
                return result
            matches, paragraphs, gaps = [], [], []
            programs = list(args.get('programs') or self.programs)
            for program in programs:
                for table in ('calls', 'copybooks', 'variables', 'paragraphs', 'files'):
                    try:
                        rows = self.rows(program, table)
                    except ToolError as error:
                        gaps.append({'program': program, 'table': table, 'error': str(error)})
                        continue
                    for row in rows:
                        name = str(row.get('name', '')).upper()
                        if name != identifier and not (table == 'files' and Path(name).stem == identifier):
                            continue
                        selected = {k: row[k] for k in ('program', '_artifact', 'name', 'caller', 'target',
                            'call_type', 'paragraph', 'parameters', 'commarea', 'line_start', 'line',
                            'source_file', 'statement', 'origin', 'controls_flow', 'evidence', 'relationships') if k in row}
                        selected['entity_type'] = table
                        matches.append({'evidence_id': self.register(selected), **selected})
                        if table == 'paragraphs':
                            paragraphs.append(self.execute('source', {'program': program,
                                'source_file': row['source_file'], 'paragraph': row['name']}))
            lookup = {'_artifact': 'verified_entity_lookup', 'identifier': identifier,
                      'programs': programs, 'matched_roles': len(matches), 'analysis_gaps': gaps,
                      'scope': 'Recorded roles, not proof of an available program implementation.'}
            return {'rows': matches, 'returned': len(matches), 'lookup_evidence_id': self.register(lookup),
                    'analysis_gaps': gaps, 'paragraph_contexts': paragraphs,
                    'limitation': 'Recorded entity roles only. A called program is not necessarily analyzed; its implementation cannot be inferred from its interface.'}
        if tool == 'quality':
            reports = []
            for program in dict.fromkeys(args['programs']):
                for table in ('quality', 'copybook_review'):
                    try:
                        for row in self.rows(program, table):
                            # Keep the findings, not a large nested dump of every
                            # comment/reference. Details remain inspectable.
                            selected = {k: row[k] for k in ('program', '_artifact',
                                'commented_out_code_count', 'unreachable_paragraphs', 'cfg_reachability',
                                'copybooks_total', 'needs_review_count', 'needs_review_copybooks',
                                'unused_copybooks_proven', 'proof_level', 'limitations') if k in row}
                            if 'commented_out_code_count' in selected:
                                selected['commented_out_code_unit'] = 'classified commented-code records; not a physical-line total'
                            reports.append({'evidence_id': self.register(selected), **selected})
                    except ToolError as error:
                        reports.append({'program': program, 'analysis_gap': str(error)})
            return {'rows': reports, 'limitation': 'Static analyzed findings; review candidates are not proven unused.'}
        if tool == 'callers':
            target = args.pop('target')
            return self.execute('query', {**args, 'programs': args.get('programs') or list(self.programs),
                'table': 'calls', 'where': [{'field': 'target', 'op': 'in' if isinstance(target, list) else 'eq', 'value': target}]})
        if tool in {'inventory', 'query', 'select', 'compare', 'compare_groups'}:
            if any(k in args for k in ('parent', 'left', 'right', 'displayed_ids')):
                raise ToolError('Internal replay fields cannot be supplied by the model.')
            grouped = tool == 'compare_groups'
            if grouped:
                saved = self.saved(args['result_id'])
                rows, _ = self.evaluate(saved['recipe'])
                key = args['group_by']
                if rows and any(field(r, key) is None for r in rows):
                    raise ToolError('Grouping field unavailable.')
                def group(value):
                    return {'tool': 'select', 'args': {'parent': saved['recipe'], 'basis': 'collection',
                        'displayed_ids': [], 'where': [{'field': key, 'op': 'eq', 'value': value}]}}
                args = {'left': group(args['left_value']), 'right': group(args['right_value']),
                        'operation': args['operation'], 'field': args.get('field', 'name'),
                        'offset': args.get('offset', 0), 'limit': args.get('limit', 100)}
                tool = 'compare'
            if tool == 'select':
                saved = self.saved(args.pop('result_id', ''))
                if args.get('basis') not in {'collection', 'displayed'}:
                    raise ToolError('Select basis must be collection or displayed.')
                args['parent'] = saved['recipe']
                args['displayed_ids'] = saved['displayed_ids']
            if tool == 'compare' and not grouped:
                args['left'] = self.saved(args.pop('left_id', ''))['recipe']
                args['right'] = self.saved(args.pop('right_id', ''))['recipe']
            offset, limit = args.pop('offset', 0), args.pop('limit', 100)
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 0 <= limit <= 100:
                raise ToolError('offset must be nonnegative; limit is 0..100 (0 returns count only).')
            recipe = {'tool': tool, 'args': args}
            rows, fingerprint = self.evaluate(recipe)
            shown = rows[offset:offset + limit]
            rid = digest((recipe, fingerprint, offset, limit))
            self.memory['collections'][rid] = {
                'recipe': recipe, 'fingerprint': fingerprint, 'total': len(rows),
                'displayed_ids': [r['_row_id'] for r in shown],
                'preview': [{'name': r.get('name'), 'program': r.get('program'), 'row_id': r['_row_id']} for r in shown[:12]]}
            self.memory['collections'] = dict(list(self.memory['collections'].items())[-8:])
            self.memory['last_result_id'] = rid
            def result_unit(recipe):
                if recipe['tool'] == 'inventory':
                    return 'analyzed programs'
                if recipe['tool'] == 'select':
                    return result_unit(recipe['args']['parent'])
                return TABLE_INFO.get(recipe['args'].get('table'), ('set members', ''))[0]
            unit = result_unit(recipe)
            relation = self.call_relation(recipe)
            summary = {'recipe': recipe, 'result_id': rid, 'unit': unit, 'total_matches': len(rows), 'offset': offset,
                       'returned': len(shown), 'complete': offset == 0 and len(shown) == len(rows),
                       'member_rows': shown,
                       'programs': sorted({r['program'] for r in rows if r.get('program')} or set(args.get('programs', []))),
                       'source_artifacts': sorted({r['_artifact'] for r in rows if r.get('_artifact')}),
                       'relation': relation, '_artifact': 'verified_collection_operation'}
            table = recipe['args'].get('table')
            summary['evidence_scope'] = {
                'variable_access': 'Recorded access sites only. Zero accesses does not prove a variable is undeclared; query variables for existence.',
                'copybooks': 'COPY inclusion. Unused/review claims require explicit needs_review/proven_unused fields and review_source; missing review fields mean unknown. Inclusion alone proves no unused status.',
                'edges': 'Static graph edges. Use source_operation/evidence for COBOL verbs; graph CALL can mean PERFORM. No runtime feasibility proof.',
                'calls': 'Recorded caller-to-target relationships within this query scope. If total_matches=0, there are no matching call parameters to report in this scope.',
            }.get(table, 'Only the recorded facts and applied predicates in this collection are established.')
            result = {'result_id': rid, 'unit': unit, 'relation': relation,
                    'offset': offset, 'evidence_scope': summary['evidence_scope'],
                    'total_matches': len(rows), 'returned': len(shown),
                    'complete': summary['complete'], 'collection_evidence_id': self.register(summary),
                    'available_fields': sorted({k for r in rows[:20] for k in r if not k.startswith('_')}),
                    'rows': [{'evidence_id': self.register({**r, 'result_id': rid, 'relation': relation}), **r} for r in shown]}
            if tool == 'query' and table == 'variable_access':
                # Bounded, explicitly labelled windows, not guessed full statements.
                # The model can expand these through source when a branch is longer.
                result['source_contexts'] = []
                for row in shown[:5]:
                    line = row.get('line_start')
                    if not isinstance(line, int) or line < 1:
                        continue
                    try:
                        window = self.execute('source_range', {'program': row['program'],
                            **({'source_file': row['source_file']} if row.get('source_file') else {}),
                            'start': max(1, line - 1), 'end': line + 8})
                        result['source_contexts'].append({'variable': row.get('variable'),
                            'access_line': line, 'window': window,
                            'limitation': 'Bounded surrounding lines, not necessarily the complete branch. Expand source if needed.'})
                    except (ToolError, OSError) as error:
                        result['source_contexts'].append({'access_line': line, 'unavailable': str(error)})
            # Keep the operation's containing paragraph available without
            # replacing the model's requested filtered result.
            if tool == 'query' and args.get('table') in {'cics', 'paragraphs'}:
                hosts = list(dict.fromkeys((r['program'], r.get('paragraph') or r.get('name')) for r in shown
                                          if r.get('paragraph') or args['table'] == 'paragraphs'))
                result['paragraph_contexts'] = [{'program': p, 'paragraph': name, 'entity_type': 'paragraph',
                    'operations': self.paragraph_operations(p, name),
                    'flow': self.paragraph_flow_context(p, name)} for p, name in hosts[:5]]
                result['paragraph_contexts_complete'] = len(hosts) <= 5
            return result
        if tool == 'inspect':
            if args.get('evidence_id'):
                row = self.evidence.get(args['evidence_id'])
                if row is None:
                    raise ToolError('Unknown evidence_id in this request.')
                value = field(row, args['field']) if args.get('field') else row
                if value is None:
                    raise ToolError('Requested evidence field unavailable.')
                return {'evidence_id': args['evidence_id'], 'data': value}
            payload = self.load(args['program'], args['artifact'])
            value = field(payload, args['field']) if args.get('field') else payload
            if value is None:
                raise ToolError('Requested artifact field unavailable.')
            offset, limit = args.get('offset', 0), args.get('limit', 8)
            if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 100:
                raise ToolError('Invalid artifact page.')
            row = {'program': args['program'], '_artifact': args['artifact'], 'field': args.get('field'),
                   'data': value[offset:offset+limit] if isinstance(value, list) else value,
                   'total': len(value) if isinstance(value, list) else 1}
            return {'evidence_id': self.register(row), **row}
        if tool == 'source':
            rows = self.rows(args['program'], 'source')
            filename = args.get('source_file') or args['program'].upper() + '.CBL'
            available = sorted({r.get('source_file', '') for r in rows})
            if filename.upper() not in {name.upper() for name in available}:
                matches = [name for name in available if Path(name).stem.upper() == filename.upper()]
                if len(matches) == 1:
                    filename = matches[0]
            rows = [r for r in rows if r.get('source_file', '').upper() == filename.upper()]
            if not rows:
                raise ToolError(f'Source member {filename!r} unavailable or ambiguous. Exact available members: {available}.')
            requested = []
            if args.get('paragraph'):
                rows = [r for r in rows if str(r.get('paragraph') or '').upper() == args['paragraph'].upper()]
                # Listing controls are not paragraph behavior. Exact physical-line
                # queries still return them verbatim for source inspection.
                rows = [r for r in rows if r.get('source_role') != 'listing_directive_not_executable']
            else:
                spans = args.get('spans', [])
                if not spans or any(not isinstance(s, list) or len(s) != 2 for s in spans):
                    raise ToolError('Source spans must be [[start,end], ...].')
                if any(type(n) is not int or n < 1 for s in spans for n in s) or any(s[1] < s[0] for s in spans):
                    raise ToolError('Invalid source span.')
                if sum(b-a+1 for a,b in spans) > 120:
                    raise ToolError('Source window exceeds 120 lines; request smaller spans.')
                requested = list(dict.fromkeys(n for a,b in spans for n in range(a,b+1)))
                rows = [r for r in rows if r['line'] in requested]
            if len(rows) > 120:
                raise ToolError('Paragraph exceeds 120 lines; read smaller source spans.')
            missing = [n for n in requested if n not in {r['line'] for r in rows}]
            return {'rows': [{'evidence_id': self.register(r), **r} for r in rows], 'returned': len(rows),
                    'paragraph_operations': self.paragraph_operations(args['program'], args['paragraph']) if args.get('paragraph') else [],
                    'paragraph_flow': self.paragraph_flow_context(args['program'], args['paragraph']) if args.get('paragraph') else {},
                    'missing_lines': missing, 'requested': args,
                    'limitation': 'Physical source records only; no absent line is inferred.'}
        if tool == 'search':
            programs = args.get('programs') or list(self.programs)
            text = str(args.get('text', '')).strip()
            if not text:
                raise ToolError('Search text must be nonempty.')
            limit = min(12, max(1, int(args.get('limit', 6))))
            if args.get('mode') == 'hybrid':
                from cobol_rag.retrieve import retrieve
                hits = []
                for p in programs:
                    self.root(p)
                    for hit in retrieve(text, self.config, top_k=min(limit, 4), program=p):
                        # Retrieved text is untrusted evidence, not instructions.
                        row = {'program': p, '_artifact': hit.metadata.get('chunk_type', 'retrieval'),
                               'source_file': hit.metadata.get('source_file'), 'text': hit.text,
                               'retrieval_metadata': hit.metadata}
                        hits.append({'evidence_id': self.register(row), **row})
                return {'rows': hits[:limit], 'returned': min(limit, len(hits)),
                        'limitation': 'Ranked hybrid retrieval; not an exhaustive inventory or absence proof.'}
            hits = []
            definitions = []
            operations = []
            for p in programs:
                source = self.rows(p, 'source')
                hits.extend(r for r in source if text.casefold() in r.get('text', '').casefold())
                body = [r for r in source if str(r.get('paragraph') or '').casefold() == text.casefold()
                        and r.get('division') == 'PROCEDURE DIVISION' and not r.get('is_comment') and not r.get('is_blank')]
                if body:
                    operations.extend(self.paragraph_operations(p, text))
                    row = {'program': p, '_artifact': 'program.source_lines.jsonl', 'paragraph': text,
                           'source_file': body[0]['source_file'],
                           'flow': self.paragraph_flow_context(p, text),
                           'definition': '\n'.join(f'{r["line"]}: {r["text"]}' for r in body[:80]),
                           'complete': len(body) <= 80}
                    definitions.append({'evidence_id': self.register(row), **row})
            return {'rows': [{'evidence_id': self.register(r), **r} for r in hits[:limit]],
                    'exact_paragraph_definitions': definitions,
                    'paragraph_operations': operations,
                    'returned': min(limit, len(hits)), 'more': len(hits) > limit,
                    'limitation': 'Literal source search, not exhaustive semantic absence; refine the term or inspect artifacts.'}
        raise ToolError('Unknown tool; use inventory, query, select, compare, inspect, source or search.')

    def recall_active(self, limit=20):
        """Revalidate the active collection without changing its saved meaning."""
        rid = self.memory.get('last_result_id')
        if not rid:
            return None
        saved = self.saved(rid)  # Reject stale corpus fingerprints.
        recipe = saved['recipe']
        rows, _ = self.evaluate(recipe)
        base = recipe
        while base['tool'] == 'select':
            base = base['args']['parent']
        unit = TABLE_INFO.get(base['args'].get('table'), ('set members', ''))[0]
        relation = self.call_relation(recipe)
        # Never offer a partial recalled list as though it were a usable set:
        # a refinement must query the whole collection, not filter a preview.
        shown = rows if len(rows) <= limit else []
        summary = dict(_artifact='verified_collection_operation', result_id=rid,
            total_matches=len(rows), returned=len(shown), complete=len(shown) == len(rows),
            unit=unit, member_rows=shown, relation=relation,
            source_artifacts=sorted({r['_artifact'] for r in rows if r.get('_artifact')}))
        return dict(result_id=rid, total_matches=len(rows), returned=len(shown),
            complete=summary['complete'], unit=unit, collection_evidence_id=self.register(summary),
            rows=[dict(r, evidence_id=self.register(dict(r, result_id=rid, relation=relation))) for r in shown])

    def context(self):
        unresolved = self.memory.get('unresolved_turn')
        active = self.memory['collections'].get(self.memory.get('last_result_id'), {})
        recipe = active.get('recipe', {})
        filters = []
        while recipe.get('tool') == 'select':
            filters.extend(recipe['args'].get('where', []))
            recipe = recipe['args']['parent']
        filters = recipe.get('args', {}).get('where', []) + filters
        exchange = deepcopy(self.memory.get('last_exchange'))
        if exchange:
            exchange['answer'] = re.sub(r'\[(?:E\d+|Source \d+)(?:\s*[,\-]\s*(?:E\d+|Source \d+))*\]', '', exchange.get('answer', ''))
        return {'evidence_scope': 'Evidence IDs expire each turn. Replay a saved result with select or query fresh evidence before citing it.',
                'unresolved_turn': deepcopy(unresolved),
                'focus_status': 'Previous successful context; latest request is unresolved.' if unresolved else 'Previous successful context.',
                'active_subject': {'programs': recipe.get('args', {}).get('programs', []),
                                   'entity_collection': recipe.get('args', {}).get('table'),
                                   'filters': filters,
                                   'unit': TABLE_INFO.get(recipe.get('args', {}).get('table'), ('set members', ''))[0],
                                   'total': active.get('total')},
                'last_result_id': self.memory.get('last_result_id'),
                'last_exchange': exchange,
                'results': {k: {'query': v['recipe'], 'total': v['total'], 'displayed': v['preview'][:5]}
                            for k, v in list(self.memory['collections'].items())[-2:]}}
