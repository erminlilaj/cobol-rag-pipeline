"""Small offline regressions for the user's collection-refinement chains."""
import json
from unittest.mock import patch

import pytest

from cobol_rag.scope import QueryScope, SessionState, EntityReference
from cobol_rag.query_plan import build_query_plan, QueryPlan, ResponseContract, validate_plan_answer
from cobol_rag.query_ir import compile_query
from cobol_rag.query import _execute_typed_query


def remember(state, plan):
    state.update(QueryScope(program=plan.program, programs=plan.programs,
                            entities=plan.entities, intent=plan.intent), [], plan.as_dict())


def follow(state, question, intent="general", program=None):
    name = program or state.current_program
    return build_query_plan(question, QueryScope(program=name, programs=(name,), intent=intent),
                            state=state)


def test_copybook_refinements_change_quality_not_subject():
    state = SessionState()
    remember(state, QueryPlan(program="PDB305", programs=("PDB305",),
                              intent="copybooks", tasks=("copybooks",)))
    plan = follow(state, "Which of those copybooks need review?", "copybooks")
    assert plan.intent == "dead_code"
    assert plan.tasks == ("review_copybooks",)
    remember(state, plan)
    plan = follow(state, "Are any of them proven unused?", "dead_code")
    assert set(plan.tasks) == {"unused_copybooks", "review_copybooks"}


def test_scope_only_followup_keeps_task_and_explicit_pivot_does_not():
    state = SessionState()
    remember(state, QueryPlan(program="PDCBVC", programs=("PDCBVC",),
                              intent="dead_code", tasks=("unreachable_code",)))
    plan = follow(state, "And in PDB305?", program="PDB305")
    assert (plan.program, plan.intent, plan.tasks) == ("PDB305", "dead_code", ("unreachable_code",))
    pivot = follow(state, "Give me an overall summary of PDB305.", "program_summary", "PDB305")
    assert pivot.intent == "program_summary"
    assert "unreachable_code" not in pivot.tasks


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / "PDB305"
    root.mkdir()
    (root / "dataflow.used_variables.json").write_text(json.dumps({
        "program": "PDB305", "variables": [
            {"variable": f"FIELD-{i}", "controls_flow": i < 7} for i in range(10)
        ]}))
    (root / "dataflow.literal_assignments.json").write_text(json.dumps({
        "program": "PDB305", "assignments": [
            {"target_variable": "WABEND-CODE", "literal": value} for value in ("A", "B", "A")
        ]}))
    with patch("cobol_rag.final_scripts_answers.find_final_scripts_root", return_value=tmp_path):
        yield


def test_control_selection_survives_presentation_change(artifacts):
    state = SessionState()
    remember(state, QueryPlan(program="PDB305", programs=("PDB305",),
                              intent="variable_inventory", tasks=("variable_inventory",),
                              response_contract=ResponseContract(format="count")))
    question = "Which of those variables control execution?"
    plan = follow(state, question, "control_flow")
    assert plan.intent == "variable_inventory"
    assert "control_usage" in plan.output_fields
    query = compile_query(question, program=plan.program, inherited_entity_type="variable",
                          output_fields=plan.output_fields)
    answer = _execute_typed_query(plan, query)
    assert "7 variable(s)" in answer and "FIELD-9" not in answer
    assert validate_plan_answer(plan, answer).passed
    remember(state, plan)
    question = "Show only the first five of them."
    plan = follow(state, question)
    query = compile_query(question, program=plan.program, inherited_entity_type="variable",
                          output_fields=plan.output_fields, allow_inventory=True)
    answer = _execute_typed_query(plan, query)
    assert "5 of 7" in answer and "FIELD-5" not in answer
    assert validate_plan_answer(plan, answer).passed


def test_distinct_count_reuses_literal_subject(artifacts):
    state = SessionState()
    entity = EntityReference(program="PDB305", entity_type="variable", value="WABEND-CODE",
                             entity_key="PDB305|VARIABLE|WABEND-CODE")
    remember(state, QueryPlan(program="PDB305", programs=("PDB305",), entities=(entity,),
                              intent="static_values", tasks=("literal_assignments",)))
    question = "How many distinct values was that?"
    plan = follow(state, question)
    assert plan.explicit_followup and plan.intent == "static_values"
    assert plan.response_contract.format == "count"
    query = compile_query(question, program=plan.program, variables=("WABEND-CODE",),
                          capability="literal_assignment")
    assert _execute_typed_query(plan, query) == "2"
    assert validate_plan_answer(plan, "2").passed
