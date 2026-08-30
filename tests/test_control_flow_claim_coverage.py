from __future__ import annotations

import unittest
from unittest.mock import patch

from cobol_rag.query_plan import _uncovered_control_flow_claims
from cobol_rag.scope import EntityReference


GRAPH = {
    "nodes": ["PARA-A", "PARA-B", "PARA-C"],
    "edges": [
        {"from": "PARA-A", "to": "PARA-B", "condition": "WS-FLAG GREATER 1"},
        {"from": "PARA-B", "to": "PARA-C"},
    ],
}


class Plan:
    def __init__(self, paragraphs=("PARA-A", "PARA-B"), program="TESTPROG"):
        self.program = program
        self.entities = tuple(
            EntityReference(program=program, entity_type="paragraph", value=p, entity_key=p)
            for p in paragraphs
        )


class ControlFlowClaimCoverageTest(unittest.TestCase):
    """Citations supporting the text is not the same as the text being true.

    An answer can cite real evidence for every sentence and still assert a
    transfer the graph does not record, or a condition taken from an unrelated
    edge. That is what produced a confident reply naming a destination which is
    not a paragraph at all.
    """

    def reasons(self, answer, plan=None):
        with patch("cobol_rag.final_scripts_answers._graph_payload", return_value=GRAPH):
            return _uncovered_control_flow_claims(plan or Plan(), answer)

    def test_a_recorded_transfer_passes(self) -> None:
        self.assertEqual(self.reasons("PARA-A -> PARA-B when WS-FLAG GREATER 1."), [])

    def test_a_transfer_the_graph_does_not_record_is_rejected(self) -> None:
        self.assertIn(
            "unrecorded_control_flow:PARA-A->PARA-C",
            self.reasons("Control flows PARA-A -> PARA-C."),
        )

    def test_a_condition_from_an_unrelated_edge_is_rejected(self) -> None:
        self.assertIn("unrecorded_condition", self.reasons("PARA-A -> PARA-B when WS-OTHER = 9."))

    def test_rendering_differences_in_a_condition_are_not_rejections(self) -> None:
        self.assertEqual(self.reasons("PARA-A -> PARA-B when WS-FLAG  GREATER   1"), [])

    def test_unconditional_transfers_are_not_condition_claims(self) -> None:
        self.assertEqual(self.reasons("PARA-B -> PARA-C unconditionally."), [])

    def test_names_that_are_not_paragraphs_are_not_treated_as_transfers(self) -> None:
        # "control flows to Yes" must not be read as a claim about a paragraph.
        self.assertEqual(
            [r for r in self.reasons("When WS-FLAG GREATER 1, control flows to Yes.")
             if r.startswith("unrecorded_control_flow")],
            [],
        )

    def test_questions_about_no_paragraph_are_left_alone(self) -> None:
        self.assertEqual(self.reasons("Anything at all.", Plan(paragraphs=())), [])


if __name__ == "__main__":
    unittest.main()
