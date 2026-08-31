from __future__ import annotations

import unittest

from cobol_rag.query import _drop_contained, _part_is_covered
from cobol_rag.query_ir import CopybookRole, CorpusReferences, compile_queries, split_question


class SplitTest(unittest.TestCase):
    """A question can ask several things whose parts need different
    capabilities. Compiling to one query answered the first and dropped the
    rest without saying so, which reads as a complete answer."""

    def test_comma_and_separates_parts(self) -> None:
        self.assertEqual(
            split_question("what is X for, and which programs use it?"),
            ["what is X for", "which programs use it"],
        )

    def test_and_before_a_question_word_separates(self) -> None:
        self.assertEqual(len(split_question("list the calls and which of them are shared")), 2)

    def test_and_between_operands_does_not_separate(self) -> None:
        """"PDCBVC and PDB305" is one subject, not two questions."""
        self.assertEqual(len(split_question("compare PDCBVC and PDB305")), 1)

    def test_a_single_question_stays_one_part(self) -> None:
        self.assertEqual(len(split_question("which paragraphs reach ABEND00?")), 1)


class CompileQueriesTest(unittest.TestCase):
    def compile(self, question):
        return compile_queries(
            question, program="PROGA", copybooks=("SHARED",),
            corpus_entity="SHARED", graph_nodes=(),
        )

    def test_each_part_compiles_separately(self) -> None:
        got = self.compile("what is SHARED for, and which other programs use it?")
        kinds = [q.kind for _part, q in got if q is not None]
        self.assertIn("copybook_role", kinds)
        self.assertIn("corpus_references", kinds)

    def test_a_part_that_compiles_to_nothing_is_kept(self) -> None:
        """Dropping it is what made a half-read question look fully answered."""
        got = self.compile("what is SHARED for, and does it run on Tuesdays?")
        self.assertTrue(any(query is None for _part, query in got))

    def test_a_single_part_question_yields_one_query(self) -> None:
        got = self.compile("what is SHARED for?")
        self.assertEqual(len(got), 1)


class CoverageTest(unittest.TestCase):
    """Reporting a part as unanswered while its answer sits above is as
    misleading as dropping it."""

    def test_a_part_whose_terms_are_present_is_covered(self) -> None:
        self.assertTrue(_part_is_covered("is it paragraphs or copybooks",
                                         "5 paragraphs unreachable; no copybooks unused"))

    def test_a_part_with_no_subject_of_its_own_is_an_elaboration(self) -> None:
        self.assertTrue(_part_is_covered("how many are there in total", "18 variables listed"))

    def test_one_shared_name_is_not_coverage(self) -> None:
        """A part sharing a single name with the answer is not answered by it."""
        self.assertFalse(_part_is_covered(
            "does PROGA pass it a literal LENGTH",
            "2 analyzed programs call X: PROGA, PROGB",
        ))


class ContainmentTest(unittest.TestCase):
    def test_a_block_whose_findings_another_states_is_dropped(self) -> None:
        small = "Shared (2):\n- A\n- B\nSource: x."
        large = "Comparison:\n- A\n- B\n- C\nSource: x."
        self.assertEqual(_drop_contained([small, large]), [large])

    def test_distinct_blocks_are_both_kept(self) -> None:
        one = "Calls:\n- PD1\nSource: x."
        two = "Copybooks:\n- CPY1\nSource: y."
        self.assertEqual(len(_drop_contained([one, two])), 2)


if __name__ == "__main__":
    unittest.main()
