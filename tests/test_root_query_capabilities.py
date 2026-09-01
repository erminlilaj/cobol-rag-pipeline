from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cobol_rag.final_scripts_answers import (
    answer_access_projection,
    answer_entity_membership,
    answer_program_comparison,
    answer_scalar_comparison,
    answer_unused_code,
    answer_semantic_projection,
)
from cobol_rag.query_ir import SemanticProjection
from cobol_rag.scope import source_addresses_in
from cobol_rag.query import _typed_query_context


class RootQueryCapabilitiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, program: str, name: str, payload: dict) -> None:
        path = self.root / program / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"program": program, **payload}), encoding="utf-8")

    def roots(self):
        return patch(
            "cobol_rag.final_scripts_answers.find_final_scripts_root",
            return_value=self.root,
        )

    def test_source_range_accepts_a_preposition(self) -> None:
        self.assertEqual(
            [(item["line_start"], item["line_end"]) for item in source_addresses_in(
                "Show lines from 404 through 406 of PDB305"
            )],
            [(404, 406)],
        )

    def test_typed_context_includes_claim_level_output_fields(self) -> None:
        plan = SimpleNamespace(
            program=None,
            programs=("A", "B"),
            entities=(),
            tasks=(),
            intent="general",
            explicit_followup=False,
            output_fields=("source_line",),
            subtasks=(SimpleNamespace(output_fields=("line_count",)),),
            entity_values_for=lambda *types: (),
        )
        context = _typed_query_context(plan, "Which has more physical lines?")
        self.assertEqual(context["output_fields"], ("source_line", "line_count"))

    def test_access_projection_never_adds_the_opposite_role(self) -> None:
        self.write("P", "dataflow.used_variables.json", {
            "variables": [
                {"variable": "READ-ME", "evidence": {
                    "read_sites": [{"paragraph": "WORK", "line_start": 10, "statement": "IF READ-ME = 1"}],
                    "write_sites": [], "read_write_sites": [],
                }},
                {"variable": "WRITE-ME", "evidence": {
                    "read_sites": [],
                    "write_sites": [{"paragraph": "WORK", "line_start": 11, "statement": "MOVE 1 TO WRITE-ME"}],
                    "read_write_sites": [],
                }},
            ],
        })
        with self.roots():
            answer = answer_access_projection("P", "WORK", "write")
        self.assertIn("WRITE-ME", answer)
        self.assertNotIn("READ-ME", answer)

    def test_named_membership_reports_each_program_and_location(self) -> None:
        self.write("A", "architecture.copybooks.json", {"content": {
            "all": ["SHARED"],
            "inclusions": [{"copybook": "SHARED", "source_file": "A.CBL", "line": 12}],
        }})
        self.write("B", "architecture.copybooks.json", {"content": {"all": [], "inclusions": []}})
        with self.roots():
            answer = answer_entity_membership(
                ("A", "B"), "SHARED", "copybook", ("exists", "source_line")
            )
        self.assertIn("A: present; A.CBL line 12", answer)
        self.assertIn("B: not present", answer)

    def test_scalar_comparison_uses_only_the_main_source_member(self) -> None:
        self.write("A", "program.source_lines.meta.json", {"files": [
            {"source_file": "A.CBL", "line_count": 100, "is_main": True},
            {"source_file": "COPY", "line_count": 900, "is_main": False},
        ]})
        self.write("B", "program.source_lines.meta.json", {"files": [
            {"source_file": "B.CBL", "line_count": 120, "is_main": True},
        ]})
        with self.roots():
            answer = answer_scalar_comparison(("A", "B"), "main_source_physical_lines")
        self.assertIn("A: 100", answer)
        self.assertIn("B: 120", answer)
        self.assertIn("B has 20 more physical lines than A", answer)

    def test_map_comparison_compares_resources_not_command_names(self) -> None:
        for program, map_name in (("A", "MAPA"), ("B", "MAPB")):
            self.write(program, "architecture.cics_operations.json", {"content": {
                "commands": ["SEND"],
                "operations": [{"command": "SEND", "statement": f"EXEC CICS SEND MAP('{map_name}') END-EXEC"}],
            }})
        with self.roots():
            answer = answer_program_comparison(("A", "B"), "map_evidence", "comparison")
        self.assertIn("Only in A (1): MAPA", answer)
        self.assertIn("Only in B (1): MAPB", answer)
        self.assertNotIn("Only in A (1): SEND", answer)

    def test_copybook_only_quality_answer_does_not_add_code_caveats(self) -> None:
        self.write("P", "architecture.unused_copybooks.json", {"content": {
            "unused_copybooks_proven": [],
            "needs_review_copybooks": ["REVIEW-ME"],
            "proof_level": "available-artifact-reference review",
        }})
        self.write("P", "quality.dead_code.json", {"content": {
            "limitations": ["commented-code limitation"],
        }})
        with self.roots():
            answer = answer_unused_code(
                "P", ("unused_copybooks", "review_copybooks")
            )
        self.assertIn("REVIEW-ME", answer)
        self.assertNotIn("commented-code limitation", answer)
        self.assertNotIn("quality.dead_code.json", answer)

    def semantic(self, **values):
        defaults = dict(
            programs=("P",), program="P", operator="project",
            capability="program_summary", entity_types=(), entity_values=(),
            fields=(), relation=None, subject_program=None, direction=None,
            source_entity=None, target_entity=None, filters=(),
        )
        defaults.update(values)
        return SemanticProjection(**defaults)

    def test_semantic_lineage_returns_only_the_bounded_path(self) -> None:
        self.write("P", "dataflow.used_variables.json", {"variables": [
            {"variable": "A", "origin": "WORKING-STORAGE", "evidence": {
                "write_sites": [], "read_sites": [
                    {"paragraph": "WORK", "line_start": 10, "statement": "MOVE A TO B."}
                ], "read_write_sites": [],
            }},
            {"variable": "B", "origin": "WORKING-STORAGE", "evidence": {
                "write_sites": [
                    {"paragraph": "WORK", "line_start": 10, "statement": "MOVE A TO B."}
                ], "read_sites": [
                    {"paragraph": "WORK", "line_start": 11, "statement": "MOVE B TO C."}
                ], "read_write_sites": [],
            }},
            {"variable": "C", "origin": "COPY:SCREEN", "evidence": {
                "write_sites": [
                    {"paragraph": "WORK", "line_start": 11, "statement": "MOVE B TO C."}
                ], "read_sites": [], "read_write_sites": [],
            }},
        ]})
        query = self.semantic(
            operator="traverse", capability="variable_lineage",
            entity_types=("variable",), entity_values=("A", "C"),
            source_entity="A", target_entity="C", direction="source_to_target",
        )
        with self.roots():
            answer = answer_semantic_projection(query)
        self.assertIn("hop 1: A -> B", answer)
        self.assertIn("hop 2: B -> C", answer)
        self.assertNotIn("Modified at", answer)

    def test_semantic_cics_projects_map_mapset_and_queue_resources(self) -> None:
        self.write("P", "architecture.cics_operations.json", {"content": {"operations": [
            {"command": "SEND", "paragraph": "SEND-MAP", "line_start": 20,
             "statement": "EXEC CICS SEND MAP('MAP1') MAPSET('SET1') END-EXEC"},
            {"command": "WRITEQ", "paragraph": "SAVE", "line_start": 30,
             "statement": "EXEC CICS WRITEQ TS QUEUE(Q-NAME) FROM(DATA) END-EXEC"},
        ]}})
        maps = self.semantic(
            capability="cics_evidence", entity_types=("map", "mapset"),
            fields=("map", "mapset", "source_line"),
        )
        queue = self.semantic(
            capability="cics_evidence", entity_types=("queue",), fields=("queue",),
            filters=(("paragraph", "eq", ("SAVE",)),),
        )
        with self.roots():
            map_answer = answer_semantic_projection(maps)
            queue_answer = answer_semantic_projection(queue)
        self.assertIn("MAP MAP1", map_answer)
        self.assertIn("MAPSET SET1", map_answer)
        self.assertIn("QUEUE Q-NAME", queue_answer)

    def test_semantic_metrics_include_requested_statement_count(self) -> None:
        self.write("P", "program.summary.json", {"meta": {
            "loc": 231, "statements": 25, "paragraphs": 19,
        }})
        self.write("P", "program.comments.json", {"metrics": {"total_lines": 850}})
        query = self.semantic(
            capability="source_metrics", operator="aggregate",
            entity_types=("metric",), fields=("line_count", "statement_count"),
        )
        with self.roots():
            answer = answer_semantic_projection(query)
        self.assertIn("physical source lines 850", answer)
        self.assertIn("executable statements 25", answer)

    def test_semantic_paragraph_count_reports_every_analyzer_observation(self) -> None:
        self.write("P", "program.summary.json", {
            "content": "P contains approximately 100 LOC and 14 paragraphs.",
            "meta": {"paragraphs": 14},
        })
        self.write("P", "controlflow.cfg.json", {"nodes": ["P", "A", "B"]})
        self.write("P", "program.comments.json", {
            "metrics": {"total_procedure_paragraphs": 4},
        })
        query = self.semantic(
            capability="source_metrics", operator="aggregate",
            entity_types=("metric",), fields=("paragraph_count",),
        )
        with self.roots():
            answer = answer_semantic_projection(query)
        for value in ("14", "3", "4"):
            self.assertIn(value, answer)
        self.assertIn("no single number is authoritative", answer)

    def test_semantic_commented_statements_respect_paragraph_filter(self) -> None:
        self.write("P", "program.comments.json", {"comments": [
            {"paragraph": "KEEP", "line": 10, "classification": "commented_out_code", "normalized_text": "MOVE A TO B."},
            {"paragraph": "OTHER", "line": 20, "classification": "commented_out_code", "normalized_text": "MOVE X TO Y."},
        ]})
        query = self.semantic(
            capability="quality_evidence", entity_types=("statement",),
            fields=("body", "source_line"),
            filters=(("paragraph", "eq", ("KEEP",)),),
        )
        with self.roots():
            answer = answer_semantic_projection(query)
        self.assertIn("MOVE A TO B", answer)
        self.assertNotIn("MOVE X TO Y", answer)


if __name__ == "__main__":
    unittest.main()
