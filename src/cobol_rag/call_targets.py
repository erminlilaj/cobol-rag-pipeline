"""Conservative source-backed resolution of variable program operands.

This is a straight-line proof, not global constant propagation. Never infer a
callee from a paragraph name, field prefix or unrelated/conditional assignment.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
import re

OPERAND = r"(?:'[^']*'|\"[^\"]*\"|[A-Z0-9][A-Z0-9-]*)"
CALL = re.compile(r'\bCALL\s+(' + OPERAND + r')(?=\s|\.|$)', re.I)
PROGRAM = re.compile(r'\bPROGRAM\s*\(\s*(' + OPERAND + r')\s*\)', re.I)
LITERAL = re.compile(r"^(['\"])([^'\"]+)\1$")
CONTROL = re.compile(r'\b(?:IF|ELSE|END-IF|EVALUATE|WHEN|END-EVALUATE|PERFORM|GO|GOTO|'
                     r'ALTER|SEARCH|READ|WRITE|REWRITE|START|RETURN|STOP|GOBACK|NEXT|EXIT|'
                     r'ON|INVALID|AT|EXCEPTION|EXEC|COPY|INCLUDE)\b', re.I)


def operand(call):
    pattern = CALL if call.get('call_type') == 'CALL' else PROGRAM
    match = pattern.search(str(call.get('statement', '')))
    return match.group(1) if match else None


def _code(row):
    text = str(row.get('text') or row.get('normalized') or '')
    if (row.get('is_comment') or row.get('source_role', '').startswith('comment')
            or text.lstrip().startswith(('*', '/'))):
        return ''
    # Retain all code, including the end of long lines; don't assume column 72
    # is a safe truncation point in a published source of unknown source format.
    if re.match(r'^[ 0-9]{6}[ */Dd-]', text):
        if text[6] in '*/Dd-':
            return '' if text[6] in '*/' else None  # Debug/continuation needs a parser.
        text = text[7:]
    # Remove free-format inline comments only outside quoted literals.
    text = re.split(r'\*>', text, maxsplit=1)[0] if not re.search(r"['\"]", text) else text
    return text.strip()


def resolve_targets(calls, source_rows):
    result = deepcopy(calls)
    for call in result:
        expression = operand(call)
        if expression is None:
            continue  # Legacy evidence without a usable statement remains unchanged.
        literal = LITERAL.fullmatch(expression)
        if literal:
            call['target_kind'] = 'literal'
            continue
        variable = expression.upper()
        call.update(target_expression=expression, target_variable=variable,
                    target_kind='unresolved_variable')
        call['target_resolution'] = {'status': 'unresolved',
            'limitation': 'No verified straight-line literal assignment at this call site. '
                          'The variable is not itself a known called-program name.'}
        line, paragraph = call.get('line_start'), call.get('paragraph')
        if type(line) is not int or not paragraph:
            continue
        files = {r.get('source_file') for r in source_rows if r.get('line') == line
                 and r.get('paragraph') == paragraph and operand({**call, 'statement': _code(r) or ''}) == expression}
        files.discard(None)
        if call.get('source_file'):
            requested = str(call['source_file']).replace('\\', '/')
            files = {f for f in files if f == requested or PurePosixPath(f).name == PurePosixPath(requested).name}
        else:
            owner = str(call.get('caller') or call.get('program') or '').upper()
            files = {f for f in files if PurePosixPath(str(f).replace('\\', '/')).stem.upper() == owner}
        if len(files) != 1:
            continue
        filename = files.pop()
        call_rows = [r for r in source_rows if r.get('source_file') == filename and r.get('line') == line
                     and r.get('paragraph') == paragraph]
        if len(call_rows) != 1:
            continue
        call_code = _code(call_rows[0])
        if not call_code or not re.match(r'^(?:CALL\b|EXEC\s+CICS\s+(?:LINK|XCTL)\b)', call_code, re.I):
            continue
        actual = operand(dict(call, statement=call_code))
        if not actual or actual.upper() != variable:
            continue
        rows = sorted((r for r in source_rows if r.get('source_file') == filename
                       and r.get('paragraph') == paragraph and type(r.get('line')) is int
                       and r['line'] < line), key=lambda r: r['line'])
        parts = []
        heading = False
        for row in rows:
            code = _code(row)
            if code is None:
                parts = []
                break
            if not code or re.fullmatch(r'(?:SKIP\d*|EJECT)\.?', code, re.I):
                continue
            if code.upper() == paragraph.upper() + '.':
                heading = True
                continue
            parts.append((row, code))
        prefix = ' '.join(code for _, code in parts)
        if not heading or not rows or set(range(rows[0]['line'], line)) - {r['line'] for r in rows}:
            continue
        # Reject branch-bearing paragraphs rather than picking the nearest
        # textual MOVE across a possible path. This intentionally leaves some
        # provable targets unresolved until a richer CFG analysis is available.
        without_literals = re.sub(r"'[^']*'|\"[^\"]*\"", "''", prefix)
        if CONTROL.search(without_literals):
            continue
        match = re.search(r'\bMOVE\s+([\'\"])([A-Z0-9_$#@-]+)\1\s+TO\s+'
                          + re.escape(variable) + r'\s*\.?\s*$', prefix, re.I)
        if not match:
            continue
        position = 0
        sites = []
        for row, code in parts:
            if position + len(code) >= match.start():
                sites.append({'source_file': filename, 'line': row['line'], 'text': row.get('text', code)})
            position += len(code) + 1
        target = match.group(2).upper()
        call.update(target=target, target_kind='resolved_variable')
        call['source_artifacts'] = ['architecture.call_parameters.json', 'program.source_lines.jsonl']
        call['target_resolution'] = {'status': 'resolved', 'method': 'straight_line_literal_move',
            'target': target, 'variable': variable, 'assignment_source': sites,
            'call_source': {'source_file': filename, 'line': line, 'statement': call['statement']},
            'limitation': 'Static call-site target, not proof the callee implementation is analyzed.'}
    return result
