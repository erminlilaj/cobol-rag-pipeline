from __future__ import annotations

import unittest

from cobol_rag.query_ir import (
    FieldProjection,
    GraphEdges,
    GraphPredicate,
    SetRelation,
    compile_query,
)


NODES = ("BROWSE-FASE1", "XCTL-LIV4", "ABEND00", "READ-TAB-SEMAF", "MAIN-PARA")


class EdgeQueryTest(unittest.TestCase):
    """A question naming both ends of an edge must compile to one query.

    The previous plan held entities in a flat bag with no roles, and the edge
    capability accepted a single paragraph, so a two-endpoint question matched
    nothing and fell through to free generation.
    """

    def compile(self, question, **kw):
        kw.setdefault("program", "PDCBVC")
        kw.setdefault("graph_nodes", NODES)
        return compile_query(question, **kw)

    def test_both_endpoints_compile_to_an_edge_query(self) -> None:
        q = self.compile(
            "under what condition does BROWSE-FASE1 go to XCTL-LIV4?",
            paragraphs=("BROWSE-FASE1", "XCTL-LIV4"),
        )
        self.assertIsInstance(q, GraphEdges)
        self.assertEqual((q.source, q.target), ("BROWSE-FASE1", "XCTL-LIV4"))

    def test_roles_follow_word_order_not_the_verb(self) -> None:
        """Paraphrases that name the same pair produce the same query."""
        for phrasing in (
            "when does BROWSE-FASE1 reach XCTL-LIV4",
            "what makes BROWSE-FASE1 transfer to XCTL-LIV4",
            "BROWSE-FASE1 branches into XCTL-LIV4 under which circumstances",
        ):
            with self.subTest(phrasing=phrasing):
                q = self.compile(phrasing, paragraphs=("BROWSE-FASE1", "XCTL-LIV4"))
                self.assertEqual((q.source, q.target), ("BROWSE-FASE1", "XCTL-LIV4"))

    def test_passive_voice_reverses_the_roles(self) -> None:
        """"A is reached from B" names A first, and B is the source.

        Assigning roles by word order alone read this question backwards and
        reported a recorded absence for an edge that exists the other way.
        """
        q = self.compile(
            "list every condition under which XCTL-LIV4 is reached from BROWSE-FASE1",
            paragraphs=("XCTL-LIV4", "BROWSE-FASE1"),
        )
        self.assertEqual((q.source, q.target), ("BROWSE-FASE1", "XCTL-LIV4"))

    def test_by_also_marks_the_source(self) -> None:
        q = self.compile(
            "when is XCTL-LIV4 called by BROWSE-FASE1?",
            paragraphs=("XCTL-LIV4", "BROWSE-FASE1"),
        )
        self.assertEqual((q.source, q.target), ("BROWSE-FASE1", "XCTL-LIV4"))

    def test_active_voice_keeps_word_order(self) -> None:
        q = self.compile(
            "when does BROWSE-FASE1 go to XCTL-LIV4?",
            paragraphs=("BROWSE-FASE1", "XCTL-LIV4"),
        )
        self.assertEqual((q.source, q.target), ("BROWSE-FASE1", "XCTL-LIV4"))

    def test_requesting_a_condition_is_recorded_in_the_projection(self) -> None:
        q = self.compile(
            "under what condition does BROWSE-FASE1 reach XCTL-LIV4?",
            paragraphs=("BROWSE-FASE1", "XCTL-LIV4"),
        )
        self.assertEqual(q.projection, "condition")

    def test_one_endpoint_is_left_to_the_existing_path(self) -> None:
        # Single-target questions already have a handler; do not divert them.
        self.assertIsNone(
            self.compile("which paragraphs lead to ABEND00", paragraphs=("ABEND00",))
        )

    def test_names_absent_from_the_graph_are_not_endpoints(self) -> None:
        self.assertIsNone(
            self.compile(
                "does WS-FLAG reach WS-OTHER", paragraphs=("WS-FLAG", "WS-OTHER")
            )
        )


class PredicateQueryTest(unittest.TestCase):
    """'Paragraphs nothing jumps to' is a property of the graph, not a name."""

    def compile(self, question):
        return compile_query(question, program="PDCBVC", graph_nodes=NODES)

    def test_negated_flow_question_compiles_to_a_predicate(self) -> None:
        for phrasing in (
            "are there any paragraphs that nothing performs or jumps to?",
            "which paragraphs are never reached?",
            "list paragraphs with no incoming calls",
        ):
            with self.subTest(phrasing=phrasing):
                q = self.compile(phrasing)
                self.assertIsInstance(q, GraphPredicate)
                self.assertEqual(q.predicate, "unreferenced")

    def test_a_flow_question_without_negation_is_not_a_predicate(self) -> None:
        self.assertNotIsInstance(self.compile("which paragraphs are reached?"), GraphPredicate)


class SetQueryTest(unittest.TestCase):
    """Naming two programs is the request to consider both.

    Relations were matched against a phrase table, so 'used by A but not by B'
    was refused over the preposition while 'only in A' was answered.
    """

    def compile(self, question, entity_type="copybook"):
        return compile_query(
            question, program="PDCBVC", programs=("PDCBVC", "PDB305"), entity_type=entity_type
        )

    def test_negation_between_the_programs_is_a_difference(self) -> None:
        for phrasing in (
            "which copybooks are used by PDCBVC but not by PDB305?",
            "copybooks in PDCBVC and not in PDB305",
            "what does PDCBVC use that PDB305 never uses?",
        ):
            with self.subTest(phrasing=phrasing):
                q = self.compile(phrasing)
                self.assertIsInstance(q, SetRelation)
                self.assertEqual(q.relation, "difference")

    def test_sameness_between_the_programs_is_an_intersection(self) -> None:
        q = self.compile("do PDCBVC and PDB305 share copybooks?")
        self.assertEqual(q.relation, "intersection")

    def test_an_unreadable_relation_still_answers_as_a_comparison(self) -> None:
        """An unrecognised phrasing must degrade to a useful answer, not a refusal."""
        q = self.compile("copybooks, PDCBVC versus PDB305, thoughts?")
        self.assertIsInstance(q, SetRelation)
        self.assertEqual(q.relation, "comparison")

    def test_a_negation_in_another_clause_is_not_the_relation(self) -> None:
        # The negation sits before both program names, not between them.
        q = self.compile("I do not recall - which copybooks do PDCBVC and PDB305 share?")
        self.assertEqual(q.relation, "intersection")


class FieldProjectionTest(unittest.TestCase):
    def test_which_copybook_declares_a_variable(self) -> None:
        q = compile_query(
            "which copybook declares TWCOB-FUNZIONE in PDCBVC?",
            program="PDCBVC",
            variables=("TWCOB-FUNZIONE",),
            graph_nodes=NODES,
        )
        self.assertIsInstance(q, FieldProjection)
        self.assertEqual((q.entity, q.field), ("TWCOB-FUNZIONE", "origin"))

    def test_a_plain_copybook_question_is_left_alone(self) -> None:
        self.assertIsNone(
            compile_query(
                "which copybooks does PDCBVC use?", program="PDCBVC", graph_nodes=NODES
            )
        )


class NoGuessTest(unittest.TestCase):
    def test_unrelated_questions_compile_to_nothing(self) -> None:
        for question in (
            "how many variables does PDCBVC have?",
            "what does PDCBVC do?",
            "show me line 226",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    compile_query(question, program="PDCBVC", graph_nodes=NODES)
                )


if __name__ == "__main__":
    unittest.main()
