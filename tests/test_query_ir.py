from __future__ import annotations

import unittest

from cobol_rag.query_ir import (
    AccessProjection,
    EntityMembership,
    FieldProjection,
    Inventory,
    GraphEdges,
    GraphPredicate,
    SetRelation,
    ScalarComparison,
    TemporalProjection,
    SemanticProjection,
    UnusedCode,
    compile_query,
)
from cobol_rag.query_plan import QuerySpecification


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

    def test_one_target_compiles_to_incoming_edges(self) -> None:
        query = self.compile("which paragraphs lead to ABEND00", paragraphs=("ABEND00",))
        self.assertIsInstance(query, GraphEdges)
        self.assertEqual((query.source, query.target, query.direction), (None, "ABEND00", "incoming"))

    def test_one_source_compiles_to_outgoing_edges(self) -> None:
        query = self.compile(
            "what transitions leave BROWSE-FASE1", paragraphs=("BROWSE-FASE1",)
        )
        self.assertIsInstance(query, GraphEdges)
        self.assertEqual(
            (query.source, query.target, query.direction),
            ("BROWSE-FASE1", None, "outgoing"),
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

    def test_exactly_one_is_a_symmetric_difference(self) -> None:
        query = self.compile(
            "Which copybooks occur in exactly one of PDCBVC and PDB305?"
        )
        self.assertEqual(query.relation, "symmetric_difference")

    def test_a_named_variable_is_membership_not_an_inventory_comparison(self) -> None:
        query = compile_query(
            "Does PDRGCODA-RETURN exist in PDB305 and PDCBVC, and where is it declared?",
            program="PDB305",
            programs=("PDB305", "PDCBVC"),
            variables=("PDRGCODA-RETURN",),
            entity_type="variable",
            operations=("exists", "locate"),
        )
        self.assertIsInstance(query, EntityMembership)
        self.assertEqual(query.entity, "PDRGCODA-RETURN")
        self.assertIn("source_line", query.fields)

    def test_physical_lines_are_a_scalar_metric(self) -> None:
        query = compile_query(
            "Which program has more physical source lines, PDB305 or PDCBVC?",
            program="PDB305",
            programs=("PDB305", "PDCBVC"),
            # The context supplies the whole recognition catalogue. None of
            # these names is explicit in the question and it must not hijack
            # the metric comparison.
            copybooks=("DFHAID", "PDRUTI01"),
            output_fields=("line_count",),
        )
        self.assertIsInstance(query, ScalarComparison)
        self.assertEqual(query.metric, "main_source_physical_lines")

    def test_physical_line_comparison_survives_a_semantic_query_spec(self) -> None:
        query = compile_query(
            "Which program has more physical source lines, PDB305 or PDCBVC?",
            program="PDB305",
            programs=("PDB305", "PDCBVC"),
            output_fields=("physical_line_count",),
            query_spec=QuerySpecification(
                operator="compare",
                capability="source_metrics",
                entity_types=("program", "metric"),
                fields=("physical_line_count",),
            ),
        )
        self.assertIsInstance(query, ScalarComparison)
        self.assertEqual(query.metric, "main_source_physical_lines")

    def test_unused_code_preserves_the_planned_quality_categories(self) -> None:
        query = compile_query(
            "Is there any unused code in PDCBVC?",
            program="PDCBVC",
            quality_categories=("commented_code", "unreachable_code"),
        )
        self.assertIsInstance(query, UnusedCode)
        self.assertEqual(query.categories, ("commented_code", "unreachable_code"))

    def test_unused_code_and_copybooks_keeps_all_requested_categories(self) -> None:
        categories = (
            "commented_code", "unreachable_code", "unused_copybooks", "review_copybooks",
        )
        query = compile_query(
            "Is there any unused code or copybook in PDCBVC?",
            program="PDCBVC",
            quality_categories=categories,
        )
        self.assertIsInstance(query, UnusedCode)
        self.assertEqual(query.categories, categories)

    def test_catalogue_copybooks_do_not_hijack_a_call_comparison(self) -> None:
        query = compile_query(
            "Which external programs are called by both PDB305 and PDCBVC?",
            program="PDB305",
            programs=("PDB305", "PDCBVC"),
            copybooks=("DFHAID", "PDRUTI01"),
            entity_type="program",
        )
        self.assertIsInstance(query, SetRelation)
        self.assertEqual(query.entity_type, "call")


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

    def test_immediate_call_context_is_temporal(self) -> None:
        query = compile_query(
            "Which condition is checked immediately after the PD1FS00 call?",
            program="PDCBVC",
            calls=("PD1FS00",),
            graph_nodes=NODES,
        )
        self.assertIsInstance(query, TemporalProjection)
        self.assertEqual((query.anchor, query.direction, query.projection), (
            "PD1FS00", "after", "condition",
        ))


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


class SemanticSpecificationTest(unittest.TestCase):
    def test_query_spec_is_authoritative_over_wording(self) -> None:
        spec = QuerySpecification(
            operator="project",
            capability="paragraph_evidence",
            entity_types=("paragraph",),
            entity_values=("XCTL-LIV4",),
            fields=("body", "outgoing_edges"),
            direction="outgoing",
        )
        results = [
            compile_query(
                wording,
                program="PDCBVC",
                programs=("PDCBVC",),
                query_spec=spec,
            )
            for wording in (
                "What does XCTL-LIV4 execute and where can it go?",
                "Inspect the body of XCTL-LIV4, then show transfers from it.",
                "This wording is deliberately unrelated to compiler phrase tables.",
            )
        ]
        self.assertTrue(all(isinstance(item, SemanticProjection) for item in results))
        self.assertTrue(all(item == results[0] for item in results[1:]))

    def test_semantic_multi_type_projection_preserves_both_types(self) -> None:
        spec = QuerySpecification(
            operator="project",
            capability="cics_evidence",
            entity_types=("map", "mapset"),
            fields=("map", "mapset", "source_line"),
        )
        query = compile_query(
            "screen resources", program="PDCBVC", programs=("PDCBVC",), query_spec=spec,
        )
        self.assertEqual(query.entity_types, ("map", "mapset"))


if __name__ == "__main__":
    unittest.main()


class UnresolvedReferenceTest(unittest.TestCase):
    """A reserved word is language; anything else that resolves to nothing is
    a name the corpus does not hold, and saying so is the answer."""

    def test_reserved_words_are_not_missing_entities(self) -> None:
        from cobol_rag.scope import COBOL_RESERVED_WORDS

        for word in ("PERFORM", "MOVE", "EVALUATE", "COMPUTE", "INITIALIZE"):
            with self.subTest(word=word):
                self.assertIn(word, COBOL_RESERVED_WORDS)

    def test_a_plausible_program_name_is_not_reserved(self) -> None:
        from cobol_rag.scope import COBOL_RESERVED_WORDS

        for name in ("PDXXXX", "PDCBVC", "PDB305", "ABEND00"):
            with self.subTest(name=name):
                self.assertNotIn(name, COBOL_RESERVED_WORDS)


class InventoryQueryTest(unittest.TestCase):
    """"Which maps does it send" and "which of them control flow" are one
    shape: a kind of thing, and a property that narrows it.

    Offered only where the planner produced no route, so it fills a gap rather
    than overriding a route that works.
    """

    def compile(self, question, **kw):
        kw.setdefault("program", "PDB305")
        kw.setdefault("graph_nodes", NODES)
        kw.setdefault("allow_inventory", True)
        return compile_query(question, **kw)

    def test_a_kind_of_thing_compiles_to_an_inventory(self) -> None:
        q = self.compile("Which BMS maps does PDB305 send to the screen?")
        self.assertIsInstance(q, Inventory)
        self.assertEqual(q.entity_type, "map")
        self.assertIsNone(q.property_filter)

    def test_a_named_property_narrows_it(self) -> None:
        q = self.compile("which variables control flow?")
        self.assertEqual((q.entity_type, q.property_filter), ("variable", "controls_flow"))

    def test_a_followup_inherits_the_kind_it_is_narrowing(self) -> None:
        # "Which of them control flow?" names no kind of its own.
        q = self.compile("Which of them control flow?", inherited_entity_type="variable")
        self.assertIsInstance(q, Inventory)
        self.assertEqual((q.entity_type, q.property_filter), ("variable", "controls_flow"))

    def test_nothing_is_offered_when_a_route_already_exists(self) -> None:
        self.assertIsNone(
            self.compile("Which BMS maps does PDB305 send?", allow_inventory=False)
        )

    def test_a_question_naming_no_kind_compiles_to_nothing(self) -> None:
        self.assertIsNone(self.compile("what happened here?"))

    def test_an_edge_question_is_not_taken_over_by_the_inventory(self) -> None:
        """Inventory runs last, so a more specific shape still wins."""
        q = self.compile(
            "under what condition does BROWSE-FASE1 reach XCTL-LIV4?",
            program="PDCBVC",
            paragraphs=("BROWSE-FASE1", "XCTL-LIV4"),
        )
        self.assertIsInstance(q, GraphEdges)

    def test_access_role_is_part_of_the_query(self) -> None:
        query = self.compile(
            "Which variables does MAIN-PARA modify?",
            variables=(),
            output_fields=("write_sites",),
        )
        self.assertIsInstance(query, AccessProjection)
        self.assertEqual(query.access_kind, "write")
