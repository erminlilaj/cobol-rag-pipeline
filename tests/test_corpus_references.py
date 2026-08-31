from __future__ import annotations

import unittest
from unittest.mock import patch

from cobol_rag.final_scripts_answers import answer_corpus_references, corpus_references
from cobol_rag.query_ir import CorpusReferences, compile_query


REGISTRY = {
    "programs": [
        {"program": "PROGA", "entities": [
            {"type": "copybook", "value": "SHARED"},
            {"type": "call", "value": "SUBPGM"},
            {"type": "paragraph", "value": "DO-WORK"},
        ]},
        {"program": "PROGB", "entities": [
            {"type": "copybook", "value": "SHARED"},
            {"type": "call", "value": "PROGA"},
        ]},
    ]
}


class CorpusReferenceTest(unittest.TestCase):
    """Some questions are about the corpus, not about one program.

    Resolving them to a single program is what made "which programs use this
    copybook" a clarification -- the copybook being in several programs is the
    answer -- and what made "which program calls PROGA" answer with PROGA's own
    outgoing calls, the only direction a program-scoped capability has.
    """

    def call(self, entity, relation=None):
        with patch("cobol_rag.final_scripts_answers._read_json", return_value=REGISTRY), \
             patch("cobol_rag.final_scripts_answers.find_final_scripts_root",
                   return_value=__import__("pathlib").Path("/x")), \
             patch("cobol_rag.final_scripts_answers.analyzed_programs",
                   return_value=("PROGA", "PROGB")):
            return answer_corpus_references(entity, relation) or ""

    def test_a_name_in_several_programs_lists_them(self) -> None:
        answer = self.call("SHARED", "includes")
        self.assertIn("PROGA", answer)
        self.assertIn("PROGB", answer)

    def test_inbound_callers_are_the_other_direction(self) -> None:
        """PROGA calls SUBPGM; PROGB calls PROGA. Asking who calls PROGA must
        not answer with what PROGA calls."""
        answer = self.call("PROGA", "calls")
        self.assertIn("PROGB", answer)
        self.assertNotIn("SUBPGM", answer)

    def test_absence_is_reported_with_the_corpus_boundary(self) -> None:
        answer = self.call("SUBPGM", "calls")
        self.assertIn("PROGA", answer)  # PROGA calls SUBPGM
        answer = self.call("DO-WORK", "calls")
        self.assertIn("No analyzed program", answer)

    def test_every_answer_states_how_much_is_analyzed(self) -> None:
        """An answer from the analyzed set is only as complete as that set, and
        at any corpus size reads as authoritative unless the bound is given."""
        for entity, relation in (("SHARED", "includes"), ("DO-WORK", "calls"), ("SHARED", None)):
            with self.subTest(entity=entity, relation=relation):
                self.assertIn("2 program(s) are analyzed", self.call(entity, relation))

    def test_subject_and_verb_agree(self) -> None:
        self.assertIn("2 analyzed programs include ", self.call("SHARED", "includes"))
        self.assertIn("1 analyzed program calls ", self.call("PROGA", "calls"))

    def test_roles_come_from_the_registry_entity_type(self) -> None:
        with patch("cobol_rag.final_scripts_answers._read_json", return_value=REGISTRY), \
             patch("cobol_rag.final_scripts_answers.find_final_scripts_root",
                   return_value=__import__("pathlib").Path("/x")):
            found = corpus_references("SHARED")
        self.assertEqual(sorted(found), ["includes"])


class CorpusCompileTest(unittest.TestCase):
    def compile(self, question):
        return compile_query(question, program="PROGA", corpus_entity="SHARED", graph_nodes=())

    def test_plural_programs_marks_a_corpus_question(self) -> None:
        for phrasing in (
            "which programs use SHARED?",
            "what programs include SHARED?",
            "how many programs use SHARED?",
            "who calls SHARED?",
        ):
            with self.subTest(phrasing=phrasing):
                self.assertIsInstance(self.compile(phrasing), CorpusReferences)

    def test_a_single_program_question_is_not_a_corpus_question(self) -> None:
        self.assertNotIsInstance(self.compile("what is SHARED for?"), CorpusReferences)


if __name__ == "__main__":
    unittest.main()
