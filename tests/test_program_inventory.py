"""A question about the corpus is answered from the corpus.

"Which programs are available?" names no program because its subject is the
set of them. Resolving scope first read the missing name as an ambiguity and
asked the user to choose a program, in answer to a request to be told which
programs exist.
"""
from __future__ import annotations

import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cobol_rag.final_scripts_answers import answer_program_inventory
from cobol_rag.query import _execute_typed_query, _program_inventory_request
from cobol_rag.query_ir import CorpusInventory, compile_query


def _ranked(capability):
    return [SimpleNamespace(capability=capability, score=0.7, margin=0.05, confident=True)]


class ProgramInventoryAnswerTest(unittest.TestCase):
    def call(self, programs, **kwargs):
        with patch("cobol_rag.final_scripts_answers.find_final_scripts_root",
                   return_value=pathlib.Path("/x")), \
             patch("cobol_rag.final_scripts_answers.analyzed_programs", return_value=programs):
            return answer_program_inventory(**kwargs)

    def test_every_analyzed_program_is_listed_with_the_count(self) -> None:
        answer = self.call(("PDB305", "PDCBVC"))
        self.assertIn("2 program(s) are analyzed", answer)
        self.assertIn("- PDB305", answer)
        self.assertIn("- PDCBVC", answer)

    def test_a_count_request_returns_the_count(self) -> None:
        self.assertEqual(self.call(("PDB305", "PDCBVC"), count_only=True), "2")

    def test_an_empty_corpus_says_so(self) -> None:
        self.assertIn("No program has been analyzed", self.call(()))

    def test_without_a_corpus_the_normal_path_answers(self) -> None:
        with patch("cobol_rag.final_scripts_answers.find_final_scripts_root", return_value=None):
            self.assertIsNone(answer_program_inventory())


class ProgramInventoryRequestTest(unittest.TestCase):
    """Plural, naming nothing, pointing at no earlier answer, requesting no evidence."""

    def ask(self, question, *, top="program_summary", tasks=()):
        with patch("cobol_rag.query._rank_question_capabilities", return_value=_ranked(top)):
            return _program_inventory_request(question, object(), SimpleNamespace(tasks=tasks))

    def test_questions_about_the_available_programs(self) -> None:
        for question in (
            "Which programs are available?",
            "Available programs",
            "What programs can I ask about?",
            "List the analyzed programs.",
            "How many programs are indexed?",
            "What programs are there?",
        ):
            with self.subTest(question=question):
                self.assertTrue(self.ask(question))

    def test_a_named_identifier_makes_it_a_question_about_that_name(self) -> None:
        self.assertFalse(self.ask("Which programs call PD0UTI01?"))

    def test_a_reference_to_an_earlier_answer_is_a_follow_up(self) -> None:
        self.assertFalse(self.ask("Which programs does it call?"))

    def test_one_program_is_not_the_set_of_programs(self) -> None:
        self.assertFalse(self.ask("What does this program do?"))

    def test_requested_evidence_is_a_restriction_a_list_cannot_satisfy(self) -> None:
        self.assertFalse(self.ask("Which programs handle pagination?", tasks=("pagination_logic",)))

    def test_the_router_separates_what_structure_reads_the_same(self) -> None:
        self.assertTrue(self.ask("Which programs are available?", top="program_summary"))
        self.assertFalse(self.ask("Which programs have unreachable code?", top="quality_evidence"))

    def test_without_a_ranking_the_question_is_left_alone(self) -> None:
        with patch("cobol_rag.query._rank_question_capabilities", side_effect=RuntimeError):
            self.assertFalse(_program_inventory_request(
                "Which programs are available?", object(), SimpleNamespace(tasks=())))


class CorpusInventoryQueryTest(unittest.TestCase):
    def test_the_router_capability_compiles_to_the_corpus(self) -> None:
        compiled = compile_query("How many programs are indexed?", program=None,
                                 capability="corpus_inventory", graph_nodes=())
        self.assertIsInstance(compiled, CorpusInventory)

    def test_a_named_entity_keeps_its_corpus_references_shape(self) -> None:
        compiled = compile_query("Which programs call PD0UTI01?", program=None,
                                 corpus_entity="PD0UTI01", capability="corpus_inventory",
                                 graph_nodes=())
        self.assertNotIsInstance(compiled, CorpusInventory)

    def test_execution_answers_from_the_registry(self) -> None:
        plan = SimpleNamespace(response_contract=SimpleNamespace(format="count"))
        with patch("cobol_rag.query.answer_program_inventory", return_value="2") as answer:
            self.assertEqual(_execute_typed_query(plan, CorpusInventory()), "2")
        answer.assert_called_once_with(count_only=True)


if __name__ == "__main__":
    unittest.main()
