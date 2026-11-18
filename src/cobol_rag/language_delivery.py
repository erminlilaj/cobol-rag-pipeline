"""Bounded, evidence-preserving Italian delivery after evidence review."""
import json
import re

from cobol_rag.query_plan import language_marker_scores


ITALIAN_FAILURE = ('Non sono riuscito a verificare una risposta completa con le prove e il tempo '
                   'disponibili. Puoi restringere la domanda o specificare il programma e il punto da esaminare.')
ITALIAN_LOCALIZATION_FAILURE = ('Ho elaborato la risposta, ma non sono riuscito a renderla in italiano '
                                'senza rischiare di modificare il codice o i dati verificati. '
                                'La risposta non viene mostrata come verificata; riprova la domanda.')


def deliver_italian(answer, budget, legacy, *, failed=False, request=None):
    """At most two model calls; never return rejected English as Italian.

    Success of evidence review and success of delivery remain separate. A failed
    localization returns a deterministic Italian notice and a failed-delivery flag.
    """
    if failed:
        return ITALIAN_FAILURE, {'output': 'localized_failure_notice', 'passed': True, 'attempts': []}
    lines, templates = legacy._separate_translation_lines(answer)
    prose = re.sub(r'\[\[P\d+\]\]', '', '\n'.join(lines))
    if not re.search(r'[A-Za-zÀ-ÿ]', prose):
        return answer, {'output': 'no_prose_to_translate', 'passed': True, 'attempts': []}
    english, italian = language_marker_scores(prose)
    if italian > english and not english:
        return answer, {'output': 'already_italian', 'passed': True, 'attempts': []}
    attempts = []
    rejected = []
    reasons = []
    for attempt in range(2):
        try:
            if attempt:
                instructions = legacy._ITALIAN_RENDER_REPAIR_PROMPT.format(
                    validation_errors='\n'.join(reasons), lines_json=json.dumps(lines, ensure_ascii=False),
                    previous_json=json.dumps(rejected, ensure_ascii=False))
            else:
                instructions = legacy._ITALIAN_RENDER_PROMPT.format(lines_json=json.dumps(lines, ensure_ascii=False))
            response = budget.call(instructions, {'resolved_request': request or {}}, reserve=0 if attempt else 1)
            rejected = response.get('lines') if isinstance(response, dict) else None
            localized = legacy._restore_translation_lines(rejected, templates)
            reasons = list(legacy._italian_translation_validation_reasons(
                answer, localized, preserve_layout=True)) if localized else ['invalid_line_or_placeholder_structure']
            if not reasons:
                # Also examine only prose slots, excluding immutable code blocks.
                localized_lines, _ = legacy._separate_translation_lines(localized)
                localized_prose = re.sub(r'\[\[P\d+\]\]', '', '\n'.join(localized_lines))
                en, it = language_marker_scores(localized_prose)
                if en or not it:
                    reasons = ['italian_prose_not_verified']
            attempts.append({'attempt': attempt + 1, 'reasons': reasons})
            if not reasons:
                return localized, {'output': 'translated' if not attempt else 'translated_after_repair',
                                   'passed': True, 'attempts': attempts}
        except Exception as error:
            reasons = [f'{type(error).__name__}: {str(error)[:300]}']
            attempts.append({'attempt': attempt + 1, 'reasons': reasons})
    return ITALIAN_LOCALIZATION_FAILURE, {'output': 'italian_localization_failed', 'passed': False,
                                        'attempts': attempts}
