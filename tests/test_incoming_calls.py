"""Incoming and outgoing calls are one relation read in two directions.

A program's own artifact records only what it calls.  Asking who calls it is
therefore not a lookup in that file but the same relation read backwards, and
losing the direction returns a true statement about the opposite fact.  These
tests hold the three places that direction can be lost: the routing gate, the
capability that executes the query, and the answer that reports the interface.
"""
from __future__ import annotations

import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cobol_rag.final_scripts_answers import (
    answer_incoming_calls,
    answer_semantic_projection,
    incoming_calls,
)
from cobol_rag.query_ir import CorpusReferences, compile_query


# PROGA calls SUBPGM with a COMMAREA and transfers to PDPRED without one;
# PROGB also calls SUBPGM.  Nothing calls PROGA -- the shape of the real
# corpus, where PD0UTI01 has two callers and PDCBVC has none.
CALLS = {
    "PROGA": {"program": "PROGA", "calls": [
        {"caller": "PROGA", "target": "SUBPGM", "call_type": "CICSLINK",
         "paragraph": "LINK-SUB", "line_start": 100,
         "commarea": "WSUB", "length": "SUB-LEN", "parameters": ["WSUB"]},
        {"caller": "PROGA", "target": "PDPRED", "call_type": "CICSXCTL",
         "paragraph": "XCTL-MAIN", "line_start": 200,
         "commarea": None, "length": None, "parameters": []},
    ]},
    "PROGB": {"program": "PROGB", "calls": [
        {"caller": "PROGB", "target": "SUBPGM", "call_type": "CALL",
         "paragraph": "DO-CALL", "line_start": 50,
         "commarea": "WOTHER", "length": None, "parameters": ["WOTHER"]},
    ]},
}


def _fake_read_json(path):
    text = str(path)
    for program, payload in CALLS.items():
        if f"/{program}/" in text:
            return payload
    return {}


class _CorpusFixture(unittest.TestCase):
    def setUp(self) -> None:
        incoming_calls.cache_clear() if hasattr(incoming_calls, "cache_clear") else None
        self._patches = [
            patch("cobol_rag.final_scripts_answers._read_json", side_effect=_fake_read_json),
            patch("cobol_rag.final_scripts_answers.find_final_scripts_root",
                  return_value=pathlib.Path("/x")),
            patch("cobol_rag.final_scripts_answers.find_program_artifact_root",
                  side_effect=lambda root, program: pathlib.Path(f"/x/{program}/")),
            patch("cobol_rag.final_scripts_answers.analyzed_programs",
                  return_value=("PROGA", "PROGB")),
        ]
        for item in self._patches:
            item.start()
            self.addCleanup(item.stop)


class IncomingCallRecordsTest(_CorpusFixture):
    def test_callers_come_from_the_callers_own_records(self) -> None:
        callers = sorted(record["caller"] for record in incoming_calls("SUBPGM"))
        self.assertEqual(callers, ["PROGA", "PROGB"])

    def test_a_program_that_calls_out_is_not_thereby_called(self) -> None:
        """PROGA calls SUBPGM and PDPRED. Asking who calls PROGA must not
        return either of them -- that is the direction inversion."""
        self.assertEqual(incoming_calls("PROGA"), ())
        answer = answer_incoming_calls("PROGA")
        self.assertIn("No analyzed program calls", answer)
        for outgoing in ("SUBPGM", "PDPRED"):
            self.assertNotIn(outgoing, answer)

    def test_the_interface_is_reported_with_the_caller(self) -> None:
        answer = answer_incoming_calls("SUBPGM")
        self.assertIn("PROGA", answer)
        self.assertIn("COMMAREA WSUB", answer)
        self.assertIn("LENGTH SUB-LEN", answer)
        self.assertIn("CICSLINK", answer)
        self.assertIn("LINK-SUB", answer)
        self.assertIn("line 100", answer)

    def test_a_call_passing_nothing_says_so(self) -> None:
        """XCTL routinely carries no COMMAREA. A recorded absence must not read
        the same as a caller the analysis failed to read."""
        answer = answer_incoming_calls("PDPRED")
        self.assertIn("PROGA", answer)
        self.assertIn("no COMMAREA or parameter recorded", answer)

    def test_absence_states_the_corpus_boundary(self) -> None:
        answer = answer_incoming_calls("NOBODY")
        self.assertIn("No analyzed program calls", answer)
        self.assertIn("2 program(s) are analyzed", answer)

    def test_every_answer_names_its_artifact(self) -> None:
        for target in ("SUBPGM", "PDPRED", "NOBODY"):
            with self.subTest(target=target):
                self.assertIn("architecture.call_parameters.json",
                              answer_incoming_calls(target))


class DirectionIsExecutedTest(_CorpusFixture):
    """The capability must read the direction it was given, not assume one."""

    def _query(self, direction):
        return SimpleNamespace(
            kind="semantic_projection", capability="call_evidence",
            operator="describe", direction=direction, entity_values=("SUBPGM",),
            relation=None, subject_program=None, fields=(), entity_types=("call",),
            programs=("PROGA",), program="PROGA", filters=(),
        )

    def test_incoming_reads_the_callers_not_the_program(self) -> None:
        with patch("cobol_rag.final_scripts_answers._semantic_program_roots",
                   return_value=[("PROGA", pathlib.Path("/x/PROGA/"))]):
            answer = answer_semantic_projection(self._query("incoming"))
        self.assertIn("PROGA", answer)
        self.assertIn("COMMAREA WSUB", answer)
        self.assertNotIn("PDPRED", answer)  # PROGA's other outgoing call

    def test_outgoing_still_uses_the_program_scoped_path(self) -> None:
        with patch("cobol_rag.final_scripts_answers._semantic_program_roots",
                   return_value=[("PROGA", pathlib.Path("/x/PROGA/"))]), \
             patch("cobol_rag.final_scripts_answers._semantic_call_answer",
                   return_value="OUTGOING") as outgoing:
            self.assertEqual(answer_semantic_projection(self._query("outgoing")), "OUTGOING")
        outgoing.assert_called_once()


class RoutingWithoutAVerbListTest(unittest.TestCase):
    """"who" asks for an agent, so the verb after it cannot be a fixed set."""

    def compile(self, question, entity="SUBPGM"):
        return compile_query(question, program="PROGA", corpus_entity=entity, graph_nodes=())

    def test_any_verb_after_who_reaches_the_corpus(self) -> None:
        for phrasing in (
            "who invokes SUBPGM?",
            "who triggers SUBPGM?",
            "who launches SUBPGM?",
            "who calls SUBPGM?",
            "who executes SUBPGM?",
        ):
            with self.subTest(phrasing=phrasing):
                self.assertIsInstance(self.compile(phrasing), CorpusReferences)

    def test_the_named_actor_still_keeps_its_own_direction(self) -> None:
        """"which programs does X call" asks the opposite question and must not
        be routed to the callers of X."""
        for phrasing in (
            "which programs does SUBPGM call?",
            "which programs are called by SUBPGM?",
            # "whom does X serve" reads as an agent question but names X as the
            # actor, so it asks what X serves -- the outgoing direction.
            "whom does SUBPGM serve?",
        ):
            with self.subTest(phrasing=phrasing):
                self.assertNotIsInstance(self.compile(phrasing), CorpusReferences)

    def test_a_purpose_question_is_not_a_call_question(self) -> None:
        self.assertNotIsInstance(self.compile("what is SUBPGM for?"), CorpusReferences)


if __name__ == "__main__":
    unittest.main()
