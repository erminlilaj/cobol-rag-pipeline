"""A literal request keeps its capability instead of widening to the inventory.

"Which variables in PDB305 receive forced literal values" resolved to the
variable inventory and returned all 105 variables: a superset that contains the
answer, conceals it, and reads as an answer, so nothing signalled that the
restriction had been dropped. Separately, a correct and complete literal answer
was discarded by the response contract for lacking a section that the literal
rendering already supplies in a stronger form.
"""
from __future__ import annotations

import unittest

from cobol_rag.query_plan import (
    _REQUESTS_LITERAL_VALUES, build_query_plan, validate_plan_answer,
)
from cobol_rag.scope import QueryScope


def plan_for(question, program="PDB305", intent="variable_inventory"):
    return build_query_plan(
        question,
        QueryScope(program=program, programs=(program,), intent=intent, confidence=0.95),
        intent=intent,
    )


class LiteralRequestIsRecognisedTest(unittest.TestCase):
    def test_the_artifacts_own_vocabulary_is_matched(self) -> None:
        for question in (
            "Which variables in PDB305 receive forced literal values?",
            "List PDB305's literal assignments and identify the variable receiving each value.",
            "which fields are hard-coded in PDB305",
            "what forced value does WABEND-CODE get",
        ):
            with self.subTest(question=question):
                self.assertTrue(_REQUESTS_LITERAL_VALUES.search(question.lower()))

    def test_an_ordinary_inventory_question_is_not_a_literal_request(self) -> None:
        for question in (
            "Which variables does PDB305 declare?",
            "list the fields in PDB305",
        ):
            with self.subTest(question=question):
                self.assertFalse(_REQUESTS_LITERAL_VALUES.search(question.lower()))


class TheCapabilitySurvivesPlanningTest(unittest.TestCase):
    def test_a_literal_request_is_not_planned_as_an_inventory(self) -> None:
        for question in (
            "Which variables in PDB305 receive forced literal values?",
            "List PDB305's literal assignments and identify the variable receiving each value.",
        ):
            with self.subTest(question=question):
                plan = plan_for(question)
                self.assertEqual(plan.intent, "static_values")
                self.assertIn("literal_assignments", plan.tasks)
                self.assertNotIn("variable_inventory", plan.tasks)

    def test_a_plain_inventory_question_still_gets_the_inventory(self) -> None:
        plan = plan_for("Which variables does PDB305 declare?")
        self.assertEqual(plan.intent, "variable_inventory")
        self.assertIn("variable_inventory", plan.tasks)
        self.assertNotIn("literal_assignments", plan.tasks)


class LiteralsDischargeTheWritesObligationTest(unittest.TestCase):
    """Every literal assignment is a write, and the literal rendering names the
    receiving variable, the value, the paragraph and the line -- strictly more
    than a write-sites list."""

    ANSWER = (
        "WABEND-CODE in PDCBVC is assigned 6 distinct literal value(s): "
        "'BR00', 'FS00', 'GET1', 'LE10', 'UT01', 'VC04'\n"
        "- PDCBVC line 226: `MOVE  'BR00'  TO  WABEND-CODE`\n"
        "- LINK-PD1VOCI line 490: `THEN MOVE  'LE10'  TO  WABEND-CODE`\n"
        "Source: `dataflow.literal_assignments.json`."
    )

    def _plan(self, tasks):
        plan = plan_for("Which literal values are moved into WABEND-CODE in PDCBVC?",
                        program="PDCBVC", intent="variable_dataflow")
        return plan.__class__(**{**plan.__dict__, "tasks": tuple(tasks)})

    def test_a_correct_literal_answer_is_not_discarded(self) -> None:
        plan = self._plan(("variable_writes", "literal_assignments"))
        result = validate_plan_answer(plan, self.ANSWER)
        self.assertNotIn("missing_requested_section:variable_writes", result.reasons)

    def test_a_writes_request_without_literals_still_needs_its_section(self) -> None:
        """The discharge is not a blanket exemption: with no literal rendering
        the obligation stands."""
        plan = self._plan(("variable_writes",))
        result = validate_plan_answer(plan, "WABEND-CODE appears in PDCBVC.")
        self.assertIn("missing_requested_section:variable_writes", result.reasons)


if __name__ == "__main__":
    unittest.main()
