"""Quality obligations survive wording, semantic refinement and execution."""
from dataclasses import replace
import json
from unittest.mock import patch

import pytest

from cobol_rag.final_scripts_answers import answer_from_final_scripts, answer_unused_code
from cobol_rag.quality_scope import QUALITY_CATEGORIES, quality_categories_named
from cobol_rag.query import _execute_typed_query, _typed_query_context
from cobol_rag.query_ir import compile_query, unused_categories_named
from cobol_rag.query_plan import (
    EvidenceSubtask, QuerySpecification, build_query_plan, merge_semantic_plan, validate_plan_answer,
)
from cobol_rag.scope import QueryScope


@pytest.fixture
def artifacts(tmp_path):
    for program in ("PDB305", "PDCBVC", "NEWPROG"):
        root = tmp_path / program
        root.mkdir()
        payloads = {
            "quality.dead_code.json": {"content": {
                "cfg_reachability": {"status": "computed_from_controlflow_cfg", "nodes_count": 3, "unreachable_nodes": ["ORPHAN"]},
                "commented_out_code": [{"line": 10, "text": "MOVE ZERO TO OLD-FIELD."}],
                "limitations": ["Static analysis is not compiler-expanded proof."],
            }},
            "architecture.unused_copybooks.json": {"content": {
                "unused_copybooks_proven": [], "needs_review_copybooks": ["REVIEW-COPY"],
                "proof_level": "available-artifact-reference review",
            }},
        }
        for name, payload in payloads.items():
            (root / name).write_text(json.dumps({"program": program, **payload}))
    with patch("cobol_rag.final_scripts_answers.find_final_scripts_root", return_value=tmp_path):
        yield tmp_path


def plan_for(question, program="PDB305"):
    return build_query_plan(question, QueryScope(
        program=program, programs=(program,), intent="dead_code", confidence=0.95,
    ), intent="dead_code")


@pytest.mark.parametrize("program", ["PDB305", "PDCBVC", "NEWPROG"])
@pytest.mark.parametrize("template", [
    "Does {program} contain unused code or unused copybooks?",
    "Which dead-code and unused-copybook findings are reported for {program}?",
    "Are there unused copybooks and dead code in {program}?",
    "Separate commented-out code, unreachable paragraphs and unused copies in {program}.",
])
def test_compound_categories_survive_every_execution_path(artifacts, program, template):
    question = template.format(program=program)
    plan = plan_for(question, program)
    # A narrower model proposal must not erase independently requested work.
    plan = merge_semantic_plan(plan, {
        "route": "technical", "intent": "dead_code",
        "tasks": ["unused_copybooks", "review_copybooks"],
    })
    assert set(plan.tasks) == set(QUALITY_CATEGORIES)
    context = _typed_query_context(plan, question)
    compiled = compile_query(question, **context)
    typed = _execute_typed_query(plan, compiled)
    direct = answer_from_final_scripts(question, intent=plan.intent, plan=plan)
    for answer in (typed, direct):
        assert "ORPHAN" in answer
        assert "MOVE ZERO TO OLD-FIELD" in answer
        assert "REVIEW-COPY" in answer
        assert "No copybook is proven unused" in answer
        assert validate_plan_answer(plan, answer).passed


@pytest.mark.parametrize("question,expected", [
    ("Any unused copybook?", QUALITY_CATEGORIES[2:]),
    ("Any unused code?", QUALITY_CATEGORIES[:2]),
    ("List unreachable paragraphs.", ("unreachable_code",)),
    ("Show commented-out code.", ("commented_code",)),
    ("Which copybooks need review?", ("review_copybooks",)),
    ("Show commented-out code and unreachable paragraphs.", QUALITY_CATEGORIES[:2]),
    ("Show unreachable code and copybooks needing review.", ("unreachable_code", "review_copybooks")),
])
def test_fallbacks_share_additive_but_narrow_category_scope(question, expected):
    assert quality_categories_named(question) == expected
    assert unused_categories_named(question) == expected
    assert plan_for(question).tasks == expected


def test_required_claim_categories_do_not_depend_on_wording(artifacts):
    plan = replace(plan_for("Inspect the analysis findings."), tasks=(), subtasks=(
        EvidenceSubtask("q1", "Inspect quality", "quality_evidence", tasks=QUALITY_CATEGORIES),
    ))
    context = _typed_query_context(plan, "Show those findings.")
    compiled = compile_query("Show those findings.", **context)
    assert set(compiled.categories) == set(QUALITY_CATEGORIES)
    answer = _execute_typed_query(plan, compiled)
    assert validate_plan_answer(plan, answer).passed


def test_semantic_spec_cannot_drop_requested_quality_categories(artifacts):
    question = "Does PDB305 contain unused code or unused copybooks?"
    plan = replace(plan_for(question), query_spec=QuerySpecification(
        operator="list", capability="quality_evidence",
        fields=("unused_copybooks", "review_copybooks"),
    ))
    compiled = compile_query(question, **_typed_query_context(plan, question))
    assert set(compiled.fields) == set(QUALITY_CATEGORIES)
    answer = _execute_typed_query(plan, compiled)
    assert "ORPHAN" in answer
    assert "MOVE ZERO TO OLD-FIELD" in answer
    assert validate_plan_answer(plan, answer).passed


def test_optional_claim_does_not_broaden_copybook_only_request(artifacts):
    question = "Is there an unused copybook in PDB305?"
    plan = replace(plan_for(question), subtasks=(
        EvidenceSubtask("optional", "Extra code review", "quality_evidence", tasks=QUALITY_CATEGORIES[:2], required=False),
    ))
    context = _typed_query_context(plan, question)
    assert context["quality_categories"] == QUALITY_CATEGORIES[2:]
    answer = _execute_typed_query(plan, compile_query(question, **context))
    assert "ORPHAN" not in answer
    assert "MOVE ZERO TO OLD-FIELD" not in answer
    assert validate_plan_answer(plan, answer).passed


def test_compound_validator_rejects_copybook_only_answer_even_with_code_citations(artifacts):
    plan = plan_for("Is there unused code or unused copybooks in PDB305?")
    answer = answer_unused_code("PDB305", QUALITY_CATEGORIES[2:])
    answer += "\nSource: `quality.dead_code.json`, `program.comments.json`."
    result = validate_plan_answer(plan, answer)
    assert not result.passed
    assert "missing_requested_section:commented_code" in result.reasons
    assert "missing_requested_section:unreachable_code" in result.reasons


@pytest.mark.parametrize("missing", [False, True])
def test_empty_and_missing_categories_are_explicit_and_distinct(artifacts, missing):
    root = artifacts / "PDB305"
    for name, content in {
        "quality.dead_code.json": {
            "cfg_reachability": {"status": "computed_from_controlflow_cfg", "unreachable_nodes": []},
            "commented_out_code": [],
        },
        "architecture.unused_copybooks.json": {"unused_copybooks_proven": [], "needs_review_copybooks": []},
    }.items():
        (root / name).write_text(json.dumps({"program": "PDB305", "content": {} if missing else content}))
    answer = answer_unused_code("PDB305", QUALITY_CATEGORIES)
    assert validate_plan_answer(plan_for("Unused code and unused copybooks in PDB305?"), answer).passed
    if missing:
        assert answer.count("unavailable") == 4
        assert "No copybook is proven unused" not in answer
    else:
        assert "unavailable" not in answer
        assert "No commented-out code" in answer
        assert "No copybooks needing review" in answer


# --------------------------------------------------------------------------
# A quantified request covers the taxonomy, not the category named first.
#
# "Summarize everything unused or unreachable" returned only unreachable
# paragraphs: the planner named one category and the resolver faithfully kept
# it. Questions that spell the categories out worked, so the coverage tracked
# the phrasing rather than the concept. Both signals needed already existed on
# the plan -- the quantifier contract that defeats truncation, and a summarize
# operation -- so neither is a new wording rule.
# --------------------------------------------------------------------------
from types import SimpleNamespace

from cobol_rag.quality_scope import quality_tasks_for_plan
from cobol_rag.query_plan import _requests_exhaustive_results


def _plan(tasks, *, result_scope="default", operations=()):
    return SimpleNamespace(
        tasks=tuple(tasks), subtasks=(), result_scope=result_scope,
        operations=tuple(operations),
    )


@pytest.mark.parametrize("question, expected", [
    ("Summarize everything unused or unreachable in PDCBVC.", True),
    ("list every paragraph", True),
    ("list all dead code", True),
    ("Which paragraphs are unreachable in PDCBVC?", False),
])
def test_everything_is_a_universal_quantifier(question, expected):
    """\\bevery\\b cannot match "everything", so the commonest phrasing of an
    exhaustive request was read as a default one and truncated."""
    assert _requests_exhaustive_results(question) is expected


def test_a_quantified_quality_request_covers_every_category():
    for plan in (
        _plan(["unreachable_code"], result_scope="all"),
        _plan(["unreachable_code", "program_summary"], operations=["summarize"]),
    ):
        assert quality_tasks_for_plan(plan) == QUALITY_CATEGORIES


def test_a_narrow_quality_request_stays_narrow():
    """One named category with no quantifier is a request for that category."""
    assert quality_tasks_for_plan(_plan(["unreachable_code"])) == ("unreachable_code",)


def test_a_request_that_names_its_categories_is_unchanged():
    named = ["commented_code", "unreachable_code", "unused_copybooks", "review_copybooks"]
    assert quality_tasks_for_plan(_plan(named)) == QUALITY_CATEGORIES


def test_a_quantifier_does_not_make_a_non_quality_request_into_one():
    """"list every variable" is exhaustive but is not a quality question, and
    must not acquire dead-code obligations from the quantifier alone."""
    plan = _plan(["variable_inventory"], result_scope="all", operations=["summarize"])
    assert quality_tasks_for_plan(plan) == ()
