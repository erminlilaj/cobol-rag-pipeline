from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cobol_rag.final_scripts_answers import (
    answer_from_final_scripts,
    program_from_question,
)
from cobol_rag.config import AppConfig
from cobol_rag.query import QueryRoutingDecision, answer_query, _deterministic_routing
from cobol_rag.query_plan import (
    QueryPlan,
    build_query_plan,
    merge_semantic_plan,
    validate_plan_answer,
)
from cobol_rag.scope import EntityReference, QueryScope, SessionState


class FinalScriptsAnswersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self._write_json(
            "architecture.copybooks.json",
            {
                "program": "PDCBVC",
                "content": {
                    "all": ["DFHAID", "PD1VOCI"],
                    "classified": {"ui_cics": ["DFHAID"], "business": ["PD1VOCI"]},
                    "inclusions": [
                        {
                            "copybook": "DFHAID",
                            "line": 194,
                            "division": "DATA DIVISION",
                            "section": "WORKING-STORAGE SECTION",
                            "statement": "COPY DFHAID SUPPRESS.",
                            "source_file": "PDCBVC.CBL",
                        },
                        {
                            "copybook": "PD1VOCI",
                            "line": 210,
                            "division": "LINKAGE DIVISION",
                            "section": "LINKAGE SECTION",
                            "statement": "COPY PD1VOCI.",
                            "source_file": "PDCBVC.CBL",
                        },
                        {
                            "copybook": "PDIABEND",
                            "line": 902,
                            "division": "PROCEDURE DIVISION",
                            "section": "UNKNOWN",
                            "statement": "COPY PDIABEND.",
                            "source_file": "PDCBVC.CBL",
                        },
                    ],
                },
            },
        )
        self._write_json(
            "dataflow.literal_assignments.json",
            {
                "program": "PDCBVC",
                "assignments": [
                    {
                        "target_variable": f"FIELD-{index:02d}",
                        "literal": str(index),
                        "literal_raw": f"'{index}'",
                        "paragraph": "TOP",
                        "line": 100 + index,
                    }
                    for index in range(30)
                ],
            },
        )
        self._write_json(
            "dataflow.variable/dataflow.variable.PD1VOCI-RETURN.json",
            {
                "program": "PDCBVC",
                "content": {
                    "variable": "PD1VOCI-RETURN",
                    "evidence": {
                        "write_sites": [
                            {
                                "paragraph": "LINK-PD1VOCI",
                                "line_start": 484,
                                "statement": "MOVE HIGH-VALUE TO PD1VOCI-RETURN.",
                            }
                        ],
                        "read_sites": [
                            {
                                "paragraph": "LINK-PD1VOCI",
                                "line_start": 489,
                                "statement": "IF PD1VOCI-RETURN EQUAL 'E'",
                            }
                        ],
                    },
                },
            },
        )
        self._write_json(
            "dataflow.variable/dataflow.variable.TWCOB-FASE.json",
            {
                "program": "PDCBVC",
                "content": {
                    "variable": "TWCOB-FASE",
                    "evidence": {
                        "write_sites": [],
                        "read_sites": [{"paragraph": "PDCBVC", "line_start": 248, "statement": "IF TWCOB-FASE = 1"}],
                    },
                },
            },
        )
        self._write_json(
            "dataflow.variable/dataflow.variable.EIBAID.json",
            {
                "program": "PDCBVC",
                "content": {
                    "variable": "EIBAID",
                    "evidence": {
                        "write_sites": [],
                        "read_sites": [{"paragraph": "BROWSE-FASE2", "line_start": 298, "statement": "IF EIBAID = DFHENTER"}],
                    },
                },
            },
        )
        self._write_json(
            "dataflow.variable/dataflow.variable.WDATE2-GG.json",
            {
                "program": "PDCBVC",
                "content": {
                    "variable": "WDATE2-GG",
                    "origin": "WORKING-STORAGE",
                    "defined_in": ["TOP"],
                    "modified_in": [],
                    "used_in": ["PREP-RIGA"],
                    "controls_flow": False,
                    "evidence": {
                        "write_sites": [],
                        "read_sites": [
                            {"paragraph": "PREP-RIGA", "line_start": 715, "statement": "MOVE WDATE2-GG TO WRIGA-GG"},
                            {"paragraph": "PREP-RIGA", "line_start": 730, "statement": "IF WDATE2-GG = ZERO"}
                        ],
                    },
                },
            },
        )
        self._write_json(
            "dataflow.variable/dataflow.variable.WABEND-CODE.json",
            {
                "program": "PDCBVC",
                "content": {
                    "variable": "WABEND-CODE",
                    "origin": "WORKING-STORAGE",
                    "defined_in": ["TOP"],
                    "modified_in": ["PDCBVC", "LINK-PD1VOCI"],
                    "used_in": ["ABEND00"],
                    "controls_flow": False,
                    "evidence": {
                        "write_sites": [
                            {"paragraph": "PDCBVC", "line_start": 226, "statement": "MOVE 'BAD PHASE' TO WABEND-CODE"},
                            {"paragraph": "LINK-PD1VOCI", "line_start": 504, "statement": "MOVE 'PD1VOCI ERROR' TO WABEND-CODE"}
                        ],
                        "read_sites": [
                            {"paragraph": "LINK-PD1VOCI", "line_start": 490, "statement": "MOVE 'PD1VOCI ERROR' TO WABEND-CODE"},
                            {"paragraph": "ABEND00", "line_start": 779, "statement": "MOVE 'ABEND' TO WABEND-CODE"}
                        ],
                    },
                },
            },
        )
        self._write_json(
            "dataflow.variable/dataflow.variable.PD1VOCI-IND.json",
            {
                "program": "PDCBVC",
                "content": {
                    "variable": "PD1VOCI-IND",
                    "origin": "WORKING-STORAGE",
                    "controls_flow": True,
                    "evidence": {
                        "write_sites": [
                            {"paragraph": "LOAD-INDEX", "line_start": 10, "statement": "MOVE 1 TO PD1VOCI-IND"},
                            {"paragraph": "NEXT-INDEX", "line_start": 20, "statement": "ADD 1 TO PD1VOCI-IND"},
                        ],
                        "read_sites": [
                            {"paragraph": "CHECK-INDEX", "line_start": 30, "statement": "IF PD1VOCI-IND > LIMIT"},
                            {"paragraph": "USE-INDEX", "line_start": 40, "statement": "MOVE TABLE-X(PD1VOCI-IND) TO X"},
                        ],
                        "control_sites": [
                            {"paragraph": "CHECK-INDEX", "line_start": 30, "statement": "IF PD1VOCI-IND > LIMIT"},
                        ],
                    },
                },
            },
        )
        for variable, evidence in (
            ("WMSGNOREC", {
                "write_sites": [],
                "read_sites": [{"paragraph": "BROWSE", "line_start": 50, "statement": "MOVE WMSGNOREC TO WKMSG"}],
                "control_sites": [],
            }),
            ("WKMSG", {
                "write_sites": [{"paragraph": "BROWSE", "line_start": 50, "statement": "MOVE WMSGNOREC TO WKMSG"}],
                "read_sites": [{"paragraph": "BROWSE", "line_start": 60, "statement": "MOVE WKMSG TO TWCOB-AREA-MSG"}],
                "control_sites": [],
            }),
            ("TWCOB-AREA-MSG", {
                "write_sites": [{"paragraph": "BROWSE", "line_start": 60, "statement": "MOVE WKMSG TO TWCOB-AREA-MSG"}],
                "read_sites": [],
                "control_sites": [],
            }),
        ):
            self._write_json(
                f"dataflow.variable/dataflow.variable.{variable}.json",
                {"program": "PDCBVC", "content": {"variable": variable, "evidence": evidence}},
            )
        self._write_json(
            "dataflow.variable/dataflow.variable.SW-FINE.json",
            {
                "program": "PDCBVC",
                "content": {
                    "variable": "SW-FINE",
                    "controls_flow": True,
                    "evidence": {
                        "write_sites": [
                            {"paragraph": "START", "line_start": 70, "statement": "MOVE '0' TO SW-FINE"},
                            {"paragraph": "FINISH", "line_start": 80, "statement": "MOVE '1' TO SW-FINE"},
                        ],
                        "read_sites": [
                            {"paragraph": "DISPLAY", "line_start": 90, "statement": "IF SW-FINE = '1'"},
                        ],
                        "control_sites": [
                            {"paragraph": "DISPLAY", "line_start": 90, "statement": "IF SW-FINE = '1'"},
                        ],
                    },
                },
            },
        )
        self._write_json(
            "business_rule/business_rule.BR-001.json",
            {
                "program": "PDCBVC",
                "content": {
                    "id": "BR-001",
                    "scope": "LINK-PD1VOCI",
                    "condition": "PD1VOCI-RETURN EQUAL 'E'",
                    "action": "JUMP -> ABEND00",
                    "evidence": {
                        "raw_evidence": "GO TO ABEND00.",
                        "source_file": "PDCBVC.CBL",
                        "line_start": 489,
                        "line_end": 490,
                    },
                },
            },
        )
        self._write_json(
            "business_rule/business_rule.BR-TWCOB.json",
            {
                "program": "PDCBVC",
                "content": {
                    "id": "BR-TWCOB",
                    "scope": "PDCBVC",
                    "condition": "NOT (TWCOB-FASE = 1 OR TWCOB-FASE = 2)",
                    "action": "JUMP -> ABEND00",
                    "evidence": {
                        "raw_evidence": "GO TO ABEND00.",
                        "source_file": "PDCBVC.CBL",
                        "line_start": 227,
                        "line_end": 227,
                    },
                },
            },
        )
        self._write_json(
            "architecture.call/architecture.call.CICSLINKBYLITERAL.PD0UTI01.json",
            {"program": "PDCBVC", "content": {"target": "PD0UTI01", "call_type": "CICSLINK"}},
        )
        self._write_json(
            "architecture.call_parameters.json",
            {
                "program": "PDCBVC",
                "calls": [
                    {
                        "target": "PD1VOCI",
                        "call_type": "CICSLINK",
                        "paragraph": "LINK-PD1VOCI",
                        "line_start": 486,
                        "parameters": ["WPD1VOCI"],
                        "commarea": "WPD1VOCI",
                    },
                    {
                        "target": "PD1FS00",
                        "call_type": "CICSLINK",
                        "paragraph": "LINK-PD1FS00",
                        "line_start": 500,
                        "statement": "EXEC CICS LINK PROGRAM('PD1FS00') COMMAREA(WPD1FS00) LENGTH(PD1FS00-LUNGH) END-EXEC.",
                        "parameters": ["WPD1FS00"],
                        "commarea": "WPD1FS00",
                        "length": "PD1FS00-LUNGH",
                        "parameter_details": [{
                            "parameter": "WPD1FS00",
                            "variables": [
                                {
                                    "variable": "PD1FS00-FUNZIONE",
                                    "writes_before_call": [{
                                        "paragraph": "PREP-LINK-PD1FS00", "line_start": 410,
                                        "statement": "MOVE '03' TO PD1FS00-FUNZIONE.",
                                    }],
                                    "reads_after_call": [],
                                },
                                {
                                    "variable": "PD1FS00-RETURN",
                                    "writes_before_call": [],
                                    "reads_after_call": [{
                                        "paragraph": "LINK-PD1FS00", "line_start": 503,
                                        "statement": "IF PD1FS00-RETURN EQUAL 'E' THEN",
                                    }],
                                },
                            ],
                        }],
                    },
                    {
                        "target": "PD0UTI01",
                        "call_type": "CICSLINK",
                        "paragraph": "LINK-PD0UTI01",
                        "line_start": 775,
                        "parameters": ["WPDRUTI01"],
                        "commarea": "WPDRUTI01",
                        "length": "400",
                    },
                    {
                        "target": "PDPRED",
                        "call_type": "CICSXCTL",
                        "paragraph": "XCTL-MAIN",
                        "line_start": 845,
                        "parameters": [],
                    }
                ],
            },
        )
        self._write_json(
            "controlflow.cfg.json",
            {
                "type": "controlflow.cfg",
                "program": "PDCBVC",
                "nodes": ["PDCBVC", "BROWSE-FASE1", "BROWSE-FASE2", "ABEND00", "RETURN-MAIN", "FINE-ELABORAZIONE"],
                "edges": [
                    {
                        "from": "PDCBVC",
                        "to": "BROWSE-FASE1",
                        "type": "JUMP",
                        "condition": "TWCOB-FASE = '1'",
                        "evidence": "GO TO BROWSE-FASE1.",
                    },
                    {
                        "from": "PDCBVC",
                        "to": "BROWSE-FASE2",
                        "type": "JUMP",
                        "condition": "TWCOB-FASE = '2'",
                        "evidence": "GO TO BROWSE-FASE2.",
                    },
                    {
                        "from": "PDCBVC",
                        "to": "ABEND00",
                        "type": "JUMP",
                        "condition": "NOT (TWCOB-FASE = '1' OR TWCOB-FASE = '2')",
                        "evidence": "GO TO ABEND00.",
                    },
                    {
                        "from": "BROWSE-FASE1",
                        "to": "RETURN-MAIN",
                        "type": "JUMP",
                        "evidence": "GO TO RETURN-MAIN.",
                    },
                ],
            },
        )
        self._write_json(
            "architecture.cics_operations.json",
            {
                "type": "architecture.cics_operations",
                "program": "PDCBVC",
                "content": {
                    "commands": ["RETURN", "SEND"],
                    "operations": [
                        {
                            "command": "SEND",
                            "paragraph": "SEND-PDCBVC1",
                            "source_file": "PDCBVC.CBL",
                            "line_start": 811,
                            "line_end": 812,
                            "statement": "EXEC CICS SEND MAP('PDCBVC1') END-EXEC",
                        },
                        {
                            "command": "RETURN",
                            "paragraph": "RETURN-MAIN",
                            "source_file": "PDCBVC.CBL",
                            "line_start": 895,
                            "line_end": 895,
                            "statement": "EXEC CICS RETURN TRANSID('PRED') END-EXEC",
                        },
                    ],
                },
            },
        )
        self._write_json(
            "program.summary.json",
            {
                "program": "PDCBVC",
                "content": "PDCBVC is a COBOL CICS program with 392 LOC and 14 paragraphs.",
            },
        )
        self._write_json(
            "program.comments.json",
            {
                "program": "PDCBVC",
                "count": 65,
                "metrics": {"total_lines": 912},
                "classification_counts": {"commented_out_code": 15},
            },
        )
        self._write_json(
            "architecture.sqlinclude/architecture.sqlinclude.SQLCA.json",
            {"program": "PDCBVC", "content": {"include": "SQLCA"}},
        )
        self._write_json(
            "architecture.sqlinclude/architecture.sqlinclude.PDWSQLER.json",
            {"program": "PDCBVC", "content": {"include": "PDWSQLER"}},
        )
        self._write_json(
            "architecture.db2_table/architecture.db2_table.DUAL.selectIntoStatement.json",
            {
                "program": "PDCBVC",
                "content": {"table": "DUAL", "stmt_type": "selectIntoStatement"},
            },
        )
        self.env = patch.dict(os.environ, {"COBOL_RAG_FINAL_SCRIPTS_DIR": str(self.root)})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.temp_dir.cleanup()

    def _write_json(self, relative_path: str, payload: dict) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_variable_catalogue_answers_whole_program_variable_questions(self) -> None:
        self._write_json(
            "dataflow.used_variables.json",
            {
                "type": "dataflow.used_variables",
                "program": "PDCBVC",
                "variables": [
                    {
                        "variable": "NPAGT",
                        "origin": "WORKING-STORAGE",
                        "controls_flow": True,
                        "defined_in": ["CALCOLA-NPAG"],
                    },
                    {
                        "variable": "ANNO-SESS",
                        "origin": "WORKING-STORAGE",
                        "controls_flow": False,
                        "defined_in": ["MUOVI-TESTATA-00"],
                    },
                ],
                "source": "pdc_var_index_used.json",
            },
        )
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="variable_inventory",
            domain="dataflow", tasks=("variable_inventory",),
            source_domains=("dataflow.used_variables",),
        )
        answer = answer_from_final_scripts(
            "name a few variables inside PDCBVC", plan=plan,
        )
        assert answer is not None
        self.assertIn("2 analyzed variable(s)", answer)
        self.assertIn("NPAGT", answer)
        self.assertIn("ANNO-SESS", answer)
        self.assertIn("dataflow.used_variables.json", answer)

    def test_variable_catalogue_returns_every_item_for_exhaustive_requests(self) -> None:
        self._write_json(
            "dataflow.used_variables.json",
            {
                "type": "dataflow.used_variables",
                "program": "PDCBVC",
                "variables": [
                    {"variable": f"FIELD-{index:03d}", "origin": "WORKING-STORAGE"}
                    for index in range(40)
                ],
                "source": "pdc_var_index_used.json",
            },
        )
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="variable_inventory",
            domain="dataflow", tasks=("variable_inventory",),
            source_domains=("dataflow.used_variables",), result_scope="all",
        )
        answer = answer_from_final_scripts("list every variable in PDCBVC", plan=plan)
        assert answer is not None
        self.assertIn("FIELD-039", answer)
        self.assertNotIn("Showing the first", answer)
        self.assertTrue(validate_plan_answer(plan, answer).passed)

    def test_program_detection_uses_programs_present_in_artifacts(self) -> None:
        question = "List the business rules implemented by PDCBVC using direct evidence."
        self.assertEqual(program_from_question(question, self.root), "PDCBVC")

    def test_exhaustive_literal_plan_returns_every_assignment(self) -> None:
        plan = QueryPlan(
            intent="static_values", tasks=("literal_assignments",),
            program="PDCBVC", programs=("PDCBVC",), result_scope="all",
        )
        answer = answer_from_final_scripts(
            "List every single forced value inside PDCBVC.", plan=plan,
        )
        assert answer is not None
        self.assertIn("30 matching item(s)", answer)
        self.assertEqual(sum(line.startswith("- line ") for line in answer.splitlines()), 30)
        self.assertNotIn("Showing the first", answer)
        self.assertIn("FIELD-29 = '29'", answer)

    def test_default_literal_plan_discloses_truncation(self) -> None:
        plan = QueryPlan(
            intent="static_values", tasks=("literal_assignments",),
            program="PDCBVC", programs=("PDCBVC",),
        )
        answer = answer_from_final_scripts("List forced values in PDCBVC.", plan=plan)
        assert answer is not None
        self.assertEqual(sum(line.startswith("- line ") for line in answer.splitlines()), 25)
        self.assertIn("Showing the first 25 of 30", answer)

    def test_literal_plan_filters_to_requested_variables(self) -> None:
        plan = QueryPlan(
            intent="static_values", tasks=("literal_assignments",),
            program="PDCBVC", programs=("PDCBVC",),
            entities=(EntityReference(
                "PDCBVC", "variable", "FIELD-07", "PDCBVC|VARIABLE|FIELD-07",
            ),),
        )
        answer = answer_from_final_scripts("What literal is assigned to FIELD-07?", plan=plan)
        assert answer is not None
        self.assertIn("1 matching item(s)", answer)
        self.assertIn("FIELD-07 = '7'", answer)
        self.assertNotIn("FIELD-06", answer)

    def test_direct_artifacts_are_selected_from_the_named_program_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory)
            for program, variable, literal in (
                ("PROGA", "A-FIELD", "A"),
                ("PROGB", "B-FIELD", "B"),
            ):
                program_root = corpus / program
                program_root.mkdir()
                (program_root / "dataflow.literal_assignments.json").write_text(
                    json.dumps({
                        "program": program,
                        "assignments": [{
                            "target_variable": variable,
                            "literal": literal,
                            "literal_raw": f"'{literal}'",
                            "paragraph": "TOP",
                            "line": 1,
                        }],
                    }),
                    encoding="utf-8",
                )
            plan = QueryPlan(
                intent="static_values", tasks=("literal_assignments",),
                program="PROGB", programs=("PROGB",), result_scope="all",
            )
            with patch.dict(os.environ, {"COBOL_RAG_FINAL_SCRIPTS_DIR": str(corpus)}):
                answer = answer_from_final_scripts(
                    "List every forced value in PROGB.", plan=plan,
                )
        assert answer is not None
        self.assertIn("B-FIELD = 'B'", answer)
        self.assertNotIn("A-FIELD", answer)

    def test_copybook_answer_supports_flat_current_layout(self) -> None:
        answer = answer_from_final_scripts("Which copybooks does PDCBVC use, and why?")
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("DFHAID", answer)
        self.assertIn("PD1VOCI", answer)
        self.assertIn("UI and CICS", answer)

    def test_llm_planner_runs_before_direct_artifact_answer(self) -> None:
        decision = QueryRoutingDecision(
            "technical", "", "copybooks", domain="integration",
            tasks=("copybook_inventory",), source_domains=("architecture.copybooks",),
            confidence=0.98,
        )
        with patch("cobol_rag.query._route_query", return_value=decision) as router:
            result = answer_query("Which copybooks does PDCBVC use, and why?", AppConfig())
        router.assert_called_once()
        self.assertEqual(result.execution_mode, "direct_artifact")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].metadata["source_file"], "architecture.copybooks.json")

    def test_exact_variable_question_uses_read_and_write_evidence(self) -> None:
        answer = answer_from_final_scripts("Which paragraphs modify and test PD1VOCI-RETURN?")
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("LINK-PD1VOCI", answer)
        self.assertIn("line 484", answer)
        self.assertIn("line 489", answer)
        self.assertIn("MOVE HIGH-VALUE", answer)
        self.assertIn("IF PD1VOCI-RETURN", answer)

    def test_compound_identifier_is_resolved_from_natural_wording(self) -> None:
        answer = answer_from_final_scripts(
            "How does PDCBVC handle the return value from PD1VOCI?"
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("PD1VOCI-RETURN", answer)
        self.assertIn("line 484", answer)
        self.assertIn("line 489", answer)

    def test_return_value_is_not_misread_as_cics_return_operation(self) -> None:
        result = answer_query(
            "How does PDCBVC handle the return value from PD1VOCI?",
            AppConfig(),
        )
        self.assertEqual(result.scope.intent, "variable_dataflow")
        self.assertIn("PD1VOCI-RETURN", result.answer)
        self.assertNotIn("RETURN", result.plan.operations)

    def test_business_rule_answer_includes_condition_action_and_paragraph(self) -> None:
        answer = answer_from_final_scripts(
            "List the business rules implemented by PDCBVC using only direct COBOL evidence. "
            "For every rule, provide the condition, action, paragraph name, variable names, "
            "and source location. Merge duplicates and do not infer business concepts that "
            "are absent from the evidence."
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("PD1VOCI-RETURN EQUAL 'E'", answer)
        self.assertIn("JUMP -> ABEND00", answer)
        self.assertIn("LINK-PD1VOCI", answer)
        self.assertIn("PDCBVC.CBL lines 489-490", answer)
        self.assertNotIn("total physical source lines", answer)

    def test_business_rule_source_line_wording_does_not_route_to_counts(self) -> None:
        answer = answer_from_final_scripts(
            "List the business rules in PDCBVC. For each rule, provide the condition, "
            "action, paragraph, variables, and source line."
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("business rules", answer)
        self.assertIn("Source location: PDCBVC.CBL lines 489-490", answer)
        self.assertNotIn("total physical source lines", answer)

    def test_file_inventory_question_lists_available_artifacts(self) -> None:
        answer = answer_from_final_scripts("what is the name of the files you have")
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("Analyzed evidence available for PDCBVC", answer)
        self.assertIn("architecture.copybooks.json", answer)
        self.assertIn("business_rule/", answer)
        self.assertNotIn("I don't have any files", answer)

    def test_control_flow_question_uses_cfg_and_terminal_evidence(self) -> None:
        answer = answer_from_final_scripts(
            "What is the control flow of PDCBVC from its entry point to termination?"
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("Entry transitions", answer)
        self.assertIn("BROWSE-FASE1", answer)
        self.assertIn("RETURN-MAIN", answer)
        self.assertIn("controlflow.cfg.json", answer)
        self.assertIn("architecture.cics_operations.json", answer)

    def test_cics_commands_have_paragraphs_and_physical_lines(self) -> None:
        answer = answer_from_final_scripts(
            "Which CICS commands does PDCBVC execute, and where is each command used?"
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("SEND in SEND-PDCBVC1", answer)
        self.assertIn("PDCBVC.CBL lines 811-812", answer)
        self.assertIn("RETURN in RETURN-MAIN", answer)
        self.assertIn("architecture.cics_operations.json", answer)

    def test_external_calls_expose_call_parameter_source(self) -> None:
        result = answer_query(
            "Which external programs does PDCBVC call, and from which paragraphs?",
            AppConfig(),
        )
        self.assertIn("PD1VOCI", result.answer)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(
            result.sources[0].metadata["source_file"],
            "architecture.call_parameters.json",
        )


    def test_call_context_plan_uses_recorded_before_and_after_sites(self) -> None:
        plan = QueryPlan(
            intent="external_programs", tasks=("call_context",), relations=("before", "after"),
            program="PDCBVC", programs=("PDCBVC",),
            entities=(EntityReference(
                "PDCBVC", "call", "PD1FS00", "PDCBVC|CALL|PD1FS00"
            ),),
        )
        answer = answer_from_final_scripts(
            "Show what happens immediately before and after the call to PD1FS00.", plan=plan,
        )
        assert answer is not None
        self.assertIn("LINK-PD1FS00 line 500", answer)
        self.assertIn("PREP-LINK-PD1FS00 line 410", answer)
        self.assertIn("MOVE '03' TO PD1FS00-FUNZIONE", answer)
        self.assertIn("LINK-PD1FS00 line 503", answer)
        self.assertIn("IF PD1FS00-RETURN EQUAL 'E'", answer)
        self.assertNotIn("PD1VOCI", answer)

    def test_multiple_variable_question_returns_each_requested_entity(self) -> None:
        answer = answer_from_final_scripts(
            "Which paragraphs test TWCOB-FASE and EIBAID?",
            intent="variable_dataflow",
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("TWCOB-FASE direct COBOL access evidence", answer)
        self.assertIn("EIBAID direct COBOL access evidence", answer)
        self.assertIn("dataflow.variable.TWCOB-FASE.json", answer)
        self.assertIn("dataflow.variable.EIBAID.json", answer)

    def test_program_summary_intent_cannot_be_overridden_by_stale_variable_text(self) -> None:
        answer = answer_from_final_scripts(
            "What is the purpose of program PDCBVC?\nResolved context: Entity: PD1VOCI-RETURN.",
            intent="program_summary",
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("technical overview", answer)
        self.assertIn("does not by itself prove a business-domain purpose", answer)
        self.assertNotIn("direct COBOL access evidence", answer)

    def test_unmatched_named_entity_does_not_return_the_whole_call_inventory(self) -> None:
        # Observed production failure: asking where WXYZ-NOTREAL is used returned
        # every outgoing call, because an unmatched requested entity produced an
        # empty filter instead of an empty match.
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="external_programs",
            domain="integration", tasks=("external_calls",),
            entities=(
                EntityReference(
                    "PDCBVC", "unknown_identifier", "WXYZ-NOTREAL",
                    "PDCBVC|UNKNOWN|WXYZ-NOTREAL",
                ),
            ),
            source_domains=("architecture.call_parameters",),
        )
        answer = answer_from_final_scripts(
            "Where is WXYZ-NOTREAL used in PDCBVC?", plan=plan,
        )
        assert answer is not None
        self.assertIn("No outgoing call evidence matched WXYZ-NOTREAL", answer)
        self.assertNotIn("PD1VOCI:", answer)

    def test_program_summary_does_not_leak_analysis_machine_paths(self) -> None:
        # The upstream artifact prose embeds the analyzer's local source path and
        # frames the program as "discovered from MAPA". Neither describes the
        # program, and the path is meaningless outside the machine that ran the scan.
        answer = answer_from_final_scripts("What is PDCBVC?", intent="program_summary")
        assert answer is not None
        self.assertNotIn("C:\\", answer)
        self.assertNotIn("discovered from MAPA", answer)
        self.assertNotIn(".CBL", answer)

    def test_program_summary_reports_disagreeing_paragraph_counts_from_both_analyzers(self) -> None:
        # MAPA and the control-flow graph disagree on PDCBVC (14 versus 65).
        # Publishing one number as fact states something the evidence does not support.
        self._write_json(
            "program.summary.json",
            {
                "program": "PDCBVC",
                "content": "PDCBVC is a COBOL CICS program discovered from MAPA.",
                "meta": {"loc": 392, "paragraphs": 14, "statements": 72},
            },
        )
        self._write_json(
            "quality.dead_code.json",
            {
                "program": "PDCBVC",
                "content": {"cfg_reachability": {"nodes_count": 65}},
            },
        )
        answer = answer_from_final_scripts("What is PDCBVC?", intent="program_summary")
        assert answer is not None
        self.assertIn("Paragraph counts differ between analyzers", answer)
        self.assertIn("14", answer)
        self.assertIn("65", answer)

    def test_control_flow_selection_uses_shared_predecessor_conditions(self) -> None:
        answer = answer_from_final_scripts(
            "How does PDCBVC decide whether to start from BROWSE-FASE1 or BROWSE-FASE2?"
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("TWCOB-FASE = '1'", answer)
        self.assertIn("TWCOB-FASE = '2'", answer)
        self.assertIn("ABEND00", answer)
        self.assertNotIn("EIBAID", answer)

    def test_copybook_location_qualifiers_return_divisions_sections_and_lines(self) -> None:
        answer = answer_from_final_scripts(
            "Which COPY statements appear in PDCBVC.CBL, and in which COBOL divisions/sections are they included?"
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("DATA DIVISION", answer)
        self.assertIn("LINKAGE SECTION", answer)
        self.assertIn("PDCBVC.CBL line 194", answer)
        self.assertIn("COPY DFHAID SUPPRESS", answer)

    def test_sql_include_only_question_excludes_tables_and_uses_stmt_type_schema(self) -> None:
        includes = answer_from_final_scripts("Which DB2 SQL includes are used by PDCBVC?")
        self.assertIsNotNone(includes)
        assert includes is not None
        self.assertIn("SQLCA", includes)
        self.assertIn("PDWSQLER", includes)
        self.assertNotIn("DB2 table DUAL", includes)
        tables = answer_from_final_scripts("Which DB2 tables are used by PDCBVC?")
        assert tables is not None
        self.assertIn("DUAL: selectIntoStatement", tables)
        qualified = answer_from_final_scripts(
            "Which DB2 tables and SQL INCLUDE members does PDCBVC use? Only include their names and source locations."
        )
        assert qualified is not None
        self.assertIn("DB2 table DUAL — artifact", qualified)
        self.assertIn("SQL include SQLCA — artifact", qualified)
        self.assertNotIn("DUAL: selectIntoStatement", qualified)

    def test_condition_effect_question_selects_business_rule_not_variable_dump(self) -> None:
        result = answer_query(
            "What happens when TWCOB-FASE is neither 1 nor 2?",
            AppConfig(),
        )
        self.assertEqual(result.scope.intent, "business_rules")
        self.assertIn("JUMP -> ABEND00", result.answer)
        self.assertIn("line 227", result.answer)
        self.assertNotIn("direct COBOL access evidence", result.answer)

    def test_copy_plan_filters_to_requested_division(self) -> None:
        result = answer_query(
            "Show only COPY statements located in the PROCEDURE DIVISION, including their exact source lines.",
            AppConfig(),
        )
        self.assertIn("PROCEDURE DIVISION", result.answer)
        self.assertNotIn("DATA DIVISION", result.answer)
        self.assertEqual(result.plan.divisions, ("PROCEDURE DIVISION",))

    def test_db2_negative_constraint_excludes_tables(self) -> None:
        result = answer_query(
            "Which SQL INCLUDE members appear in PDCBVC? Do not include DB2 tables or unrelated dependencies.",
            AppConfig(),
        )
        self.assertIn("SQLCA", result.answer)
        self.assertNotIn("DB2 table DUAL", result.answer)
        self.assertIn("db2_table", result.plan.exclude_types)

    def test_call_target_and_followup_are_filtered_to_one_entity(self) -> None:
        state = SessionState()
        first = answer_query("Where does PDCBVC call PD0UTI01?", AppConfig(), session_state=state)
        state.update(first.scope, [], plan=first.plan.as_dict())
        self.assertIn("PD0UTI01", first.answer)
        self.assertNotIn("PD1VOCI:", first.answer)
        second = answer_query("What parameters are passed to that call?", AppConfig(), session_state=state)
        self.assertIn("WPDRUTI01", second.answer)
        self.assertNotIn("WPD1VOCI", second.answer)

    def test_previous_variable_qualifier_is_preserved_for_program_summary(self) -> None:
        state = SessionState(
            current_program="PDCBVC", current_entity_type="variable",
            current_entity_value="PD1VOCI-RETURN",
            current_entity_key="PDCBVC|VARIABLE|PD1VOCI-RETURN",
            current_intent="variable_dataflow",
        )
        decision = QueryRoutingDecision(
            "technical", "", "program_summary",
            category="multi_source_synthesis", operations=("summarize",),
            source_domains=("program.summary", "dataflow.variable"), confidence=0.96,
        )
        with patch("cobol_rag.query._route_query", return_value=decision):
            second = answer_query(
                "Give me an overall technical summary of PDCBVC focusing on the previously discussed variable.",
                AppConfig(), session_state=state,
            )
        self.assertEqual(second.route, "technical")
        self.assertEqual(second.scope.entity_value, "PD1VOCI-RETURN")
        self.assertIn("Requested variable focus:", second.answer)

    def test_cics_link_and_xctl_are_operations_not_entity_prefixes(self) -> None:
        result = answer_query(
            "List every CICS LINK or XCTL issued by PDCBVC, with the target program, paragraph, COMMAREA, and line.",
            AppConfig(),
        )
        self.assertFalse(result.scope.ambiguous)
        self.assertEqual(set(result.plan.operations), {"LINK", "XCTL"})
        self.assertIn("PD0UTI01", result.answer)
        self.assertIn("PDPRED", result.answer)
        self.assertNotIn("PXRSEMAF", result.answer)

    def test_elongated_greeting_is_left_to_the_llm_router(self) -> None:
        self.assertIsNone(_deterministic_routing("good morninggg", QueryScope(), None))

    def test_semantic_intent_dispatches_control_flow_paraphrase(self) -> None:
        question = "Walk me through what happens after PDCBVC starts until it hands control away or ends."
        with (
            patch(
                "cobol_rag.query._route_query",
                return_value=QueryRoutingDecision("technical", "", "control_flow"),
            ),
            patch("cobol_rag.query.retrieve") as retrieve,
        ):
            result = answer_query(question, AppConfig())
        retrieve.assert_not_called()
        self.assertIn("Entry transitions", result.answer)
        self.assertEqual(
            [source.metadata["source_file"] for source in result.sources],
            ["controlflow.cfg.json", "architecture.cics_operations.json"],
        )

    def test_semantic_intent_dispatches_cics_paraphrase(self) -> None:
        question = "Show the transaction-processing instructions issued by PDCBVC and their code locations."
        with (
            patch(
                "cobol_rag.query._route_query",
                return_value=QueryRoutingDecision("technical", "", "cics_operations"),
            ),
            patch("cobol_rag.query.retrieve") as retrieve,
        ):
            result = answer_query(question, AppConfig())
        retrieve.assert_not_called()
        self.assertIn("SEND in SEND-PDCBVC1", result.answer)
        self.assertEqual(result.sources[0].metadata["source_file"], "architecture.cics_operations.json")

    def test_semantic_intent_dispatches_business_rule_paraphrase(self) -> None:
        question = "Describe the decisions PDCBVC makes and the resulting actions, with direct code evidence."
        with (
            patch(
                "cobol_rag.query._route_query",
                return_value=QueryRoutingDecision("technical", "", "business_rules"),
            ),
            patch("cobol_rag.query.retrieve") as retrieve,
        ):
            result = answer_query(question, AppConfig())
        retrieve.assert_not_called()
        self.assertIn("PD1VOCI-RETURN EQUAL 'E'", result.answer)

    def test_generic_variable_description_uses_exact_artifact(self) -> None:
        decision = QueryRoutingDecision(
            "technical", "", "variable_dataflow",
            category="single_source", operations=("describe",),
            source_domains=("dataflow.variable",), confidence=0.96,
        )
        with patch("cobol_rag.query._route_query", return_value=decision):
            result = answer_query("What is WDATE2-GG?", AppConfig())
        self.assertEqual(result.execution_mode, "direct_artifact")
        self.assertEqual(result.plan.planner_source, "hybrid_llm")
        self.assertIn("Origin: WORKING-STORAGE", result.answer)
        self.assertIn("Used in: PREP-RIGA", result.answer)
        self.assertIn("line 715", result.answer)

    def test_variable_description_corrects_assignment_sites_mislabeled_as_reads(self) -> None:
        decision = QueryRoutingDecision(
            "technical", "", "variable_dataflow",
            operations=("describe",), source_domains=("dataflow.variable",), confidence=0.95,
        )
        with patch("cobol_rag.query._route_query", return_value=decision):
            result = answer_query("What is WABEND-CODE?", AppConfig())
        self.assertIn("line 490", result.answer)
        self.assertIn("line 779", result.answer)
        self.assertIn("rendered as writes", result.answer)
        modified, tested = result.answer.split("Tested/read at:", 1)
        self.assertIn("line 490", modified)
        self.assertNotIn("line 490", tested)

    def test_multi_variable_comparison_is_not_a_stored_answer(self) -> None:
        decision = QueryRoutingDecision(
            "technical", "", "variable_dataflow",
            category="multi_source_comparison", operations=("compare",),
            source_domains=("dataflow.variable",), requires_comparison=True, confidence=0.97,
        )
        with patch("cobol_rag.query._route_query", return_value=decision):
            result = answer_query("Compare WDATE2-GG and WABEND-CODE in PDCBVC.", AppConfig())
        self.assertEqual(result.execution_mode, "direct_artifact")
        self.assertEqual(result.plan.category, "multi_source_comparison")
        self.assertEqual(set(result.scope.entity_values), {"WDATE2-GG", "WABEND-CODE"})
        self.assertIn("Comparison of direct variable evidence", result.answer)
        self.assertIn("dataflow.variable.WDATE2-GG.json", result.answer)
        self.assertIn("dataflow.variable.WABEND-CODE.json", result.answer)

    def test_exact_variable_scope_rejects_incompatible_semantic_program_flow(self) -> None:
        entity = EntityReference(
            "PDCBVC", "variable", "SW-FINE", "PDCBVC|VARIABLE|SW-FINE",
        )
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            intent="control_flow",
        )
        base = build_query_plan(
            "Which values are assigned to SW-FINE, and how does each value affect subsequent control flow?",
            scope,
            intent="control_flow",
        )
        merged = merge_semantic_plan(base, {
            "intent": "control_flow",
            "tasks": ["complete_program_flow"],
            "source_domains": ["controlflow.cfg"],
            "operations": ["trace"],
        })
        self.assertEqual(merged.intent, "variable_dataflow")
        self.assertNotIn("complete_program_flow", merged.tasks)
        answer = answer_from_final_scripts(
            "Which values are assigned to SW-FINE, and how does each value affect subsequent control flow?",
            plan=merged,
        )
        assert answer is not None
        self.assertIn("MOVE '0' TO SW-FINE", answer)
        self.assertIn("MOVE '1' TO SW-FINE", answer)
        self.assertIn("Control-flow use:", answer)
        self.assertIn("IF SW-FINE = '1'", answer)
        self.assertIn("'1' is directly tested", answer)
        self.assertIn("no direct control condition for that value", answer)

    def test_exhaustive_variable_sites_report_complete_coverage(self) -> None:
        entity = EntityReference(
            "PDCBVC", "variable", "PD1VOCI-IND", "PDCBVC|VARIABLE|PD1VOCI-IND",
        )
        plan = QueryPlan(
            intent="variable_dataflow", tasks=("variable_reads", "variable_writes"),
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            result_scope="all", output_fields=("source_line", "exact_statement"),
        )
        answer = answer_from_final_scripts(
            "List every paragraph that modifies or reads PD1VOCI-IND.", plan=plan,
        )
        assert answer is not None
        self.assertIn("Write coverage: 2/2", answer)
        self.assertIn("Read coverage: 2/2", answer)
        self.assertIn("line 10", answer)
        self.assertIn("line 40", answer)

    def test_variable_lineage_expands_across_intermediate_artifacts(self) -> None:
        entity = EntityReference(
            "PDCBVC", "variable", "WMSGNOREC", "PDCBVC|VARIABLE|WMSGNOREC",
        )
        plan = QueryPlan(
            intent="variable_dataflow",
            tasks=("variable_definition", "variable_reads", "variable_writes", "variable_lineage"),
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            result_scope="all",
        )
        answer = answer_from_final_scripts(
            "Trace WMSGNOREC through every intermediate variable using direct evidence.",
            plan=plan,
        )
        assert answer is not None
        self.assertIn("WMSGNOREC -> WKMSG", answer)
        self.assertIn("WKMSG -> TWCOB-AREA-MSG", answer)
        self.assertIn("dataflow.variable.WKMSG.json", answer)
        self.assertIn("dataflow.variable.TWCOB-AREA-MSG.json", answer)

    def test_source_metrics_line_question_satisfies_output_contract(self) -> None:
        decision = QueryRoutingDecision(
            "technical", "", "source_metrics", tasks=("source_metrics",),
            source_domains=("program.summary", "program.comments"), confidence=0.98,
        )
        with patch("cobol_rag.query._route_query", return_value=decision):
            result = answer_query("How many lines is PDCBVC?", AppConfig())
        self.assertEqual(result.execution_mode, "direct_artifact")
        self.assertIn("912 total physical source lines", result.answer)
        self.assertNotIn("did not satisfy", result.answer)

    def test_source_metrics_survives_a_vague_semantic_router_update(self) -> None:
        question = "How many lines of code is the file PDCBVC?"
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), intent="general",
        )
        base = build_query_plan(question, scope, intent="general")
        self.assertEqual(base.intent, "source_metrics")
        self.assertIn("source_metrics", base.tasks)
        self.assertIn("line_count", base.output_fields)
        merged = merge_semantic_plan(base, {
            "intent": "program_summary",
            "tasks": ["program_summary"],
            "source_domains": ["program.summary"],
            "output_fields": ["source_line"],
        })
        self.assertEqual(merged.intent, "source_metrics")
        self.assertEqual(merged.tasks, ("source_metrics",))
        answer = answer_from_final_scripts(question, plan=merged)
        assert answer is not None
        self.assertIn("912 total physical source lines", answer)
        self.assertIn("392 LOC", answer)
        self.assertTrue(validate_plan_answer(merged, answer).passed)

    def test_variable_control_outcome_joins_business_rule(self) -> None:
        entity = EntityReference(
            "PDCBVC", "variable", "PD1VOCI-RETURN",
            "PDCBVC|VARIABLE|PD1VOCI-RETURN",
        )
        plan = QueryPlan(
            intent="variable_dataflow",
            tasks=("variable_reads", "variable_writes", "control_outcome"),
            relations=("condition_causes",),
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
        )
        answer = answer_from_final_scripts(
            "Trace PD1VOCI-RETURN and explain how it affects control flow.", plan=plan,
        )
        assert answer is not None
        self.assertIn("Resulting control-flow actions:", answer)
        self.assertIn("PD1VOCI-RETURN EQUAL 'E'", answer)
        self.assertIn("JUMP -> ABEND00", answer)
        self.assertIn("business_rule.BR-001.json", answer)

    def test_variable_call_option_usage_joins_full_call_statement(self) -> None:
        self._write_json(
            "dataflow.variable/dataflow.variable.PD1FS00-LUNGH.json",
            {
                "program": "PDCBVC",
                "content": {
                    "variable": "PD1FS00-LUNGH",
                    "evidence": {
                        "write_sites": [],
                        "read_sites": [{
                            "paragraph": "LINK-PD1FS00", "line_start": 501,
                            "statement": "LENGTH(PD1FS00-LUNGH) END-EXEC.",
                        }],
                        "control_sites": [],
                    },
                },
            },
        )
        entity = EntityReference(
            "PDCBVC", "variable", "PD1FS00-LUNGH",
            "PDCBVC|VARIABLE|PD1FS00-LUNGH",
        )
        plan = QueryPlan(
            intent="variable_dataflow", tasks=("variable_reads", "call_option_usage"),
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            output_fields=("exact_statement", "source_line"),
        )
        answer = answer_from_final_scripts(
            "How does PD1FS00-LUNGH control the CICS LINK? Include the exact statement.",
            plan=plan,
        )
        assert answer is not None
        self.assertIn("CICS/call option usage:", answer)
        self.assertIn("supplies the LENGTH option", answer)
        self.assertIn("COMMAREA(WPD1FS00)", answer)
        self.assertIn("lines 500-501", answer)

    def test_variable_composition_collects_child_assignments(self) -> None:
        for variable, evidence in (
            ("ROW-GROUP", {
                "write_sites": [{"paragraph": "BUILD-ROW", "line_start": 100, "statement": "MOVE SPACES TO ROW-GROUP"}],
                "read_sites": [{"paragraph": "SEND-ROW", "line_start": 110, "statement": "MOVE ROW-GROUP TO MAPO"}],
                "control_sites": [],
            }),
            ("ROW-A", {
                "write_sites": [{"paragraph": "BUILD-ROW", "line_start": 101, "statement": "MOVE SOURCE-A TO ROW-A"}],
                "read_sites": [], "control_sites": [],
            }),
            ("ROW-B", {
                "write_sites": [{"paragraph": "BUILD-ROW", "line_start": 102, "statement": "MOVE SOURCE-B TO ROW-B"}],
                "read_sites": [], "control_sites": [],
            }),
        ):
            self._write_json(
                f"dataflow.variable/dataflow.variable.{variable}.json",
                {"program": "PDCBVC", "content": {"variable": variable, "evidence": evidence}},
            )
        entity = EntityReference(
            "PDCBVC", "variable", "ROW-GROUP", "PDCBVC|VARIABLE|ROW-GROUP",
        )
        plan = QueryPlan(
            intent="variable_dataflow",
            tasks=("variable_reads", "variable_writes", "variable_composition"),
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
        )
        answer = answer_from_final_scripts(
            "How is ROW-GROUP constructed and transferred?", plan=plan,
        )
        assert answer is not None
        self.assertIn("Construction/context evidence:", answer)
        self.assertIn("MOVE SOURCE-A TO ROW-A", answer)
        self.assertIn("MOVE SOURCE-B TO ROW-B", answer)
        self.assertIn("MOVE ROW-GROUP TO MAPO", answer)

    def test_terminal_lineage_reports_overwrites_and_nearby_branch(self) -> None:
        self._write_json(
            "dataflow.variable/dataflow.variable.WKMSG.json",
            {
                "program": "PDCBVC",
                "content": {
                    "variable": "WKMSG", "origin": "WORKING-STORAGE",
                    "evidence": {
                        "write_sites": [{
                            "paragraph": "BROWSE", "line_start": 50,
                            "statement": "MOVE WMSGNOREC TO WKMSG",
                        }],
                        "read_sites": [
                            {"paragraph": "BROWSE", "line_start": 55, "statement": "MOVE WKERR1 TO WKMSG"},
                            {"paragraph": "BROWSE", "line_start": 60, "statement": "MOVE WKMSG TO TWCOB-AREA-MSG"},
                        ],
                        "control_sites": [],
                    },
                },
            },
        )
        self._write_json(
            "business_rule/business_rule.BR-MSG.json",
            {
                "program": "PDCBVC",
                "content": {
                    "id": "BR-MSG", "scope": "BROWSE", "condition": "COUNT = 0",
                    "action": "JUMP -> NEXT-PROGRAM",
                    "evidence": {"line_start": 61, "raw_evidence": "GO TO NEXT-PROGRAM"},
                },
            },
        )
        entity = EntityReference(
            "PDCBVC", "variable", "WMSGNOREC", "PDCBVC|VARIABLE|WMSGNOREC",
        )
        plan = QueryPlan(
            intent="variable_dataflow",
            tasks=("variable_lineage", "lineage_terminal"),
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
        )
        answer = answer_from_final_scripts(
            "Trace WMSGNOREC toward the displayed message through every intermediate variable.",
            plan=plan,
        )
        assert answer is not None
        self.assertIn("Terminal destination: TWCOB-AREA-MSG", answer)
        self.assertIn("WKERR1 -> WKMSG", answer)
        self.assertIn("JUMP -> NEXT-PROGRAM", answer)
        self.assertIn("no further MOVE to a screen/map field is proven", answer)


if __name__ == "__main__":
    unittest.main()
