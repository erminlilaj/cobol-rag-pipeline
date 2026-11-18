"""Strict evidence citation groups; ranges never invent registered evidence."""
import re


GROUP = re.compile(r'\[(E\d+(?:\s*(?:-|,)\s*E?\d+)*)\]')


def ids_in_group(group):
    ids = []
    for item in group.split(','):
        numbers = re.fullmatch(r'\s*E?(\d+)(?:\s*-\s*E?(\d+))?\s*', item)
        if not numbers:
            raise ValueError('Invalid evidence citation group.')
        first, last = int(numbers[1]), int(numbers[2] or numbers[1])
        if first < 1 or last < first or last - first > 199:
            raise ValueError('Invalid or oversized evidence citation range.')
        ids.extend(f'E{number}' for number in range(first, last + 1))
    return ids


def evidence_ids(text):
    if any(not GROUP.fullmatch(match[0]) for match in re.finditer(r'\[E(?=\d)[^\]\n]*\]', text)):
        raise ValueError('Malformed evidence citation group.')
    return [eid for match in GROUP.finditer(text) for eid in ids_in_group(match[1])]


def render_citations(text, selected):
    labels = {eid: f'[Source {index}]' for index, eid in enumerate(selected, 1)}
    def render(match):
        ids = ids_in_group(match[1])
        if any(eid not in labels for eid in ids):
            raise ValueError('Citation range includes evidence outside the selected sources.')
        return ' '.join(labels[eid] for eid in ids)
    return GROUP.sub(render, text)
