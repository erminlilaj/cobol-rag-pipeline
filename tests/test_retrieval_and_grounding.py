from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock, patch

from cobol_rag.chat import ChatSession, ChatTurn
from cobol_rag.config import AppConfig
from cobol_rag.loaders.rag_documents import RagDocumentsLoader
from cobol_rag.capability_router import (
    VARIABLE_ASPECT_DESCRIPTORS,
    CapabilityMatch,
    confident_aspects,
    eligible_capabilities,
    rank_capabilities,
)
from cobol_rag.query import (
    _build_prompt,
    _build_constrained_routing_prompt,
    _build_routing_prompt,
    _absent_capability_request,
    _attempt_evidence_subtask,
    _entity_contract_failed,
    _rerouted_subtask,
    EvidenceSubtaskResult,
    _decision_respects_candidates,
    _routing_candidates,
    _chunk_types_for_plan,
    _complete_with_transient_retry,
    _conversational_route_is_blocked,
    _deterministic_routing,
    _direct_handler_supports,
    _ensure_citations,
    _parse_routing_decision,
    _retrieve_for_plan,
    _route_query,
    _resolve_weak_technical_route,
    _routing_conflicts_with_verified_scope,
    _try_structured_plan_answer,
    _semantic_route_hint,
    answer_query,
    QueryAnswer,
    QueryRoutingDecision,
)
from cobol_rag.query_plan import (
    _CAPABILITY_ENTITY_TYPES,
    _CAPABILITY_TASKS,
    EvidenceSubtask,
    QueryPlan,
    build_query_plan,
    detect_message_language,
    derive_evidence_subtasks,
    merge_semantic_plan,
    plan_for_subtask,
    resolve_response_language,
    validate_plan_answer,
)
from cobol_rag.final_scripts_answers import absent_capability_answer
from cobol_rag.retrieve import (
    EvidenceGuard, RetrievalOutcome, RetrievalResult, _deduplicate_results, _detect_intent,
)
from cobol_rag.scope import EntityReference, QueryScope, SessionState


class MetadataLoaderTest(unittest.TestCase):
    def test_entity_and_evidence_metadata_survive_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rag_documents.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "variable-1",
                        "program": "PDCBVC",
                        "type": "dataflow.variable",
                        "text": "PD1VOCI-RETURN is modified and tested.",
                        "metadata": {
                            "variable": "PD1VOCI-RETURN",
                            "paragraph": "LINK-PD1VOCI",
                            "intent_domain": "variable_dataflow",
                            "entity_type": "variable",
                            "entity_key": "PDCBVC|VARIABLE|PD1VOCI-RETURN",
                            "source_system": "mapa_hamza",
                            "evidence_path": "programs/PDCBVC/artifacts/dataflow.variable.json",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            document = RagDocumentsLoader(AppConfig()).load(source)[0].document
            for key in (
                "variable",
                "paragraph",
                "intent_domain",
                "entity_type",
                "entity_key",
                "source_system",
                "evidence_path",
            ):
                self.assertIn(key, document.metadata)


class RetrievalPolicyTest(unittest.TestCase):
    def test_detects_business_rule_and_variable_intents(self) -> None:
        self.assertEqual(_detect_intent("List the business rules in PDCBVC"), "business_rules")
        self.assertEqual(
            _detect_intent("Which paragraphs test PD1VOCI-RETURN?"),
            "variable_dataflow",
        )
        self.assertEqual(
            _detect_intent("Trace execution from the entry point to termination"),
            "control_flow",
        )
        self.assertEqual(
            _detect_intent("Where does the program execute its CICS commands?"),
            "cics_operations",
        )

    def test_duplicate_nodes_from_one_source_are_collapsed(self) -> None:
        results = [
            RetrievalResult(0.9, "first", {"source_id": "same", "chunk_id": "same"}),
            RetrievalResult(0.8, "second", {"source_id": "same", "chunk_id": "same"}),
            RetrievalResult(0.7, "third", {"source_id": "other", "chunk_id": "other"}),
        ]
        deduplicated = _deduplicate_results(results)
        self.assertEqual([item.text for item in deduplicated], ["first", "third"])


class PromptGroundingTest(unittest.TestCase):
    def test_compound_variable_and_call_request_becomes_independent_subtasks(self) -> None:
        variables = (
            "PXCSEMAF-REQ", "PXCSEMAF-NAME", "PXCSEMAF-AGENT",
            "PXCSEMAF-CALLER", "PXCSEMAF-CALLER-TYPE",
        )
        entities = (
            EntityReference("PDCBVC", "call", "PXRSEMAF", "PDCBVC|PXRSEMAF|CALL"),
            *(
                EntityReference("PDCBVC", "variable", value, f"PDCBVC|VARIABLE|{value}")
                for value in variables
            ),
        )
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), entities=entities,
            intent="variable_dataflow",
        )
        plan = build_query_plan(
            "Before PXRSEMAF is called, what literal values are assigned to "
            + ", ".join(variables)
            + "? Include source lines.",
            scope,
            intent="variable_dataflow",
        )

        capabilities = {subtask.capability for subtask in plan.subtasks}
        self.assertIn("literal_assignment", capabilities)
        self.assertIn("call_context", capabilities)
        call_subtask = next(item for item in plan.subtasks if item.capability == "call_context")
        literal_subtask = next(item for item in plan.subtasks if item.capability == "literal_assignment")
        self.assertEqual(call_subtask.entity_values, ("PXRSEMAF",))
        self.assertEqual(literal_subtask.entity_values, variables)
        self.assertEqual(call_subtask.relations, ("before",))
        self.assertNotIn("before", literal_subtask.relations)

        literal_plan = plan_for_subtask(plan, literal_subtask)
        self.assertEqual(literal_plan.intent, "static_values")
        self.assertEqual(literal_plan.entity_values, variables)
        self.assertNotIn("PXRSEMAF", literal_plan.entity_values)

    def test_literal_subtask_subsumes_same_entity_write_subtask(self) -> None:
        entity = EntityReference(
            "PDCBVC", "variable", "WABEND-CODE", "PDCBVC|VARIABLE|WABEND-CODE",
        )
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="variable_dataflow",
            domain="dataflow", tasks=("variable_writes", "literal_assignments"),
            entities=(entity,), source_domains=("dataflow.variable", "dataflow.literal_assignments"),
        )
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "variable_dataflow",
            "tasks": ["variable_writes", "literal_assignments"],
            "subtasks": [
                {
                    "capability": "variable_access", "tasks": ["variable_writes"],
                    "entity_values": ["WABEND-CODE"], "relations": ["before"],
                },
                {
                    "capability": "literal_assignment", "tasks": ["literal_assignments"],
                    "entity_values": ["WABEND-CODE"], "relations": ["before"],
                },
            ],
        })

        self.assertEqual([item.capability for item in merged.subtasks], ["literal_assignment"])
        self.assertEqual(merged.subtasks[0].relations, ())

    def test_semantic_subtasks_cannot_invent_entities_and_keep_fallback_coverage(self) -> None:
        entity = EntityReference(
            "PDCBVC", "variable", "WCTRIG", "PDCBVC|VARIABLE|WCTRIG",
        )
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="variable_dataflow",
            domain="dataflow", tasks=("variable_reads", "variable_writes"),
            entities=(entity,), source_domains=("dataflow.variable",),
        )
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "variable_dataflow",
            "tasks": ["variable_reads", "variable_writes"],
            "subtasks": [{
                "description": "Verify reads",
                "capability": "variable_access",
                "tasks": ["variable_reads"],
                "entity_values": ["WCTRIG", "INVENTED-FIELD"],
            }],
        })

        self.assertTrue(all(
            "INVENTED-FIELD" not in subtask.entity_values for subtask in merged.subtasks
        ))
        covered = {task for subtask in merged.subtasks for task in subtask.tasks}
        self.assertEqual(covered, {"variable_reads", "variable_writes"})

    def test_answer_generation_retries_one_transient_model_failure(self) -> None:
        llm = Mock()
        expected = Mock(text="supported answer")
        llm.complete.side_effect = [RuntimeError("temporary Ollama failure"), expected]
        response = _complete_with_transient_retry(llm, "prompt")
        self.assertIs(response, expected)
        self.assertEqual(llm.complete.call_count, 2)

    def test_conversational_route_bypasses_artifacts_and_retrieval(self) -> None:
        with (
            patch(
                "cobol_rag.query._route_query",
                return_value=QueryRoutingDecision("conversational", "Hello there!"),
            ),
            patch("cobol_rag.query.answer_from_final_scripts", return_value=None) as artifact_answer,
            patch("cobol_rag.query.retrieve") as retrieve,
        ):
            result = answer_query("a conversational message", AppConfig())

        self.assertEqual(result.answer, "Hello there!")
        self.assertEqual(result.route, "conversational")
        self.assertEqual(result.sources, [])
        artifact_answer.assert_not_called()
        retrieve.assert_not_called()

    def test_simple_greeting_is_left_to_the_llm_router(self) -> None:
        self.assertIsNone(_deterministic_routing("hi", QueryScope(), None))

    def test_current_message_language_overrides_stale_session_language(self) -> None:
        italian_state = SessionState(response_language="it")
        english_plan = build_query_plan("hi", QueryScope(), state=italian_state)
        self.assertEqual(english_plan.response_language, "en")
        self.assertEqual(english_plan.response_language_source, "message")

        english_state = SessionState(response_language="en")
        italian_plan = build_query_plan("Ciao, come stai?", QueryScope(), state=english_state)
        self.assertEqual(italian_plan.response_language, "it")
        self.assertEqual(italian_plan.response_language_source, "message")

    def test_explicit_language_request_overrides_the_message_language(self) -> None:
        self.assertEqual(
            resolve_response_language("Can you answer in Italian?", SessionState()),
            ("it", "explicit_request"),
        )
        self.assertEqual(
            resolve_response_language("Puoi rispondere in inglese?", SessionState(response_language="it")),
            ("en", "explicit_request"),
        )
        self.assertEqual(detect_message_language("Why are you replying in Italian?"), "en")

    def test_identifier_only_message_preserves_the_session_language(self) -> None:
        language, source = resolve_response_language(
            "PD1VOCI-RETURN?", SessionState(response_language="it"),
        )
        self.assertEqual((language, source), ("it", "session"))

    def test_semantic_merge_cannot_override_resolved_message_language(self) -> None:
        plan = build_query_plan("How are you?", QueryScope(), state=SessionState(response_language="it"))
        merged = merge_semantic_plan(plan, {"response_language": "it"})
        self.assertEqual(merged.response_language, "en")
        self.assertEqual(merged.response_language_source, "message")

    def test_router_prompt_makes_current_language_contract_authoritative(self) -> None:
        plan = build_query_plan("hi", QueryScope(), state=SessionState(response_language="it"))
        prompt = _build_routing_prompt(
            "hi", None, SessionState(response_language="it"), preliminary_plan=plan,
        )
        self.assertIn("Required response language: en (resolved from: message)", prompt)
        self.assertIn("clear English message therefore switches", prompt)

    def test_wrong_language_router_reply_is_retried(self) -> None:
        wrong = type("Response", (), {"text": json.dumps({
            "route": "conversational", "category": "conversational",
            "domain": "conversation", "intent": "general", "tasks": [],
            "relations": [], "response_language": "en",
            "reply": "Certo, tu? In che cosa posso aiutare? (English: Hello, how can I help you?)",
        })})()
        recovered = type("Response", (), {"text": "<reply>Hello! How are you?</reply>"})()
        plan = build_query_plan("hi", QueryScope(), state=SessionState(response_language="it"))
        with patch("cobol_rag.query.build_llm") as build:
            build.return_value.complete.side_effect = [wrong, recovered]
            decision = _route_query(
                "hi", AppConfig(), session_state=SessionState(response_language="it"),
                preliminary_plan=plan, preliminary_scope=QueryScope(),
            )
        self.assertEqual(build.call_count, 2)
        self.assertEqual(decision.route, "conversational")
        self.assertEqual(decision.response_language, "en")
        self.assertEqual(decision.reply, "Hello! How are you?")

    def test_router_parses_structured_model_output(self) -> None:
        decision = _parse_routing_decision(
            '```json\n{"route":"technical","intent":"control_flow","reply":"should be discarded"}\n```'
        )
        self.assertEqual(decision, QueryRoutingDecision("technical", "", "control_flow"))

    def test_conversational_router_ignores_nontechnical_intent_label(self) -> None:
        decision = _parse_routing_decision(json.dumps({
            "route": "conversational", "category": "single_source",
            "domain": "general", "intent": "language_preference",
            "response_language": "it", "reply": "Sì, posso rispondere in italiano.",
        }))
        self.assertEqual(decision.route, "conversational")
        self.assertEqual(decision.intent, "general")
        self.assertEqual(decision.domain, "conversation")
        self.assertEqual(decision.response_language, "it")
        self.assertEqual(decision.reply, "Sì, posso rispondere in italiano.")

    def test_router_parses_hybrid_plan_category_and_domains(self) -> None:
        decision = _parse_routing_decision(
            json.dumps({
                "route": "technical",
                "category": "multi_source_comparison",
                "intent": "variable_dataflow",
                "operations": ["compare"],
                "source_domains": ["dataflow.variable"],
                "output_fields": ["origin", "read_sites"],
                "requires_comparison": True,
                "confidence": 0.94,
                "reply": "",
            })
        )
        self.assertEqual(decision.category, "multi_source_comparison")
        self.assertEqual(decision.operations, ("compare",))
        self.assertEqual(decision.source_domains, ("dataflow.variable",))
        self.assertEqual(decision.output_fields, ("origin", "read_sites"))
        self.assertTrue(decision.requires_comparison)
        self.assertEqual(decision.planner_source, "semantic_llm")

    def test_router_parses_hierarchical_tasks_relations_language_and_exclusions(self) -> None:
        decision = _parse_routing_decision(json.dumps({
            "route": "technical", "category": "multi_source_synthesis",
            "domain": "integration", "intent": "external_programs",
            "tasks": ["call_context"], "relations": ["before", "after"],
            "operations": ["show_context"], "excluded_operations": ["XCTL"],
            "source_domains": ["architecture.call_parameters"],
            "output_fields": ["source_line", "exact_statement"],
            "response_language": "it", "confidence": 0.97, "reply": "",
        }))
        self.assertEqual(decision.domain, "integration")
        self.assertEqual(decision.tasks, ("call_context",))
        self.assertEqual(decision.relations, ("before", "after"))
        self.assertEqual(decision.excluded_operations, ("XCTL",))
        self.assertEqual(decision.response_language, "it")

    def test_relational_and_non_english_plans_bypass_fixed_handlers(self) -> None:
        relational = QueryPlan(
            intent="control_flow", tasks=("paragraph_references", "paragraph_body"),
            relations=("referenced_by", "contains"),
        )
        italian = QueryPlan(intent="copybooks", tasks=("copybook_inventory",), response_language="it")
        self.assertFalse(_direct_handler_supports(relational))
        self.assertFalse(_direct_handler_supports(italian))
        self.assertTrue(_direct_handler_supports(QueryPlan(intent="copybooks", tasks=("copybook_inventory",))))

    def test_task_plan_selects_parent_and_child_artifact_families(self) -> None:
        plan = QueryPlan(
            intent="control_flow", tasks=("paragraph_references", "paragraph_body"),
            source_domains=("controlflow.cfg",),
        )
        chunk_types = set(_chunk_types_for_plan(plan) or [])
        self.assertIn("controlflow.cfg", chunk_types)
        self.assertIn("integration.paragraph_context", chunk_types)
        self.assertIn("paragraph_logic", chunk_types)

    def test_literal_exclusion_is_removed_before_semantic_routing(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="external_programs")
        plan = build_query_plan(
            "Show only LINK calls in PDCBVC; exclude XCTL.", scope,
            intent="external_programs",
        )
        self.assertIn("LINK", plan.operations)
        self.assertNotIn("XCTL", plan.operations)
        self.assertEqual(plan.excluded_operations, ("XCTL",))

    def test_exhaustive_literal_request_is_preserved_as_a_plan_contract(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="static_values")
        plan = build_query_plan(
            "List every single forced value inside PDCBVC.", scope,
            intent="static_values",
        )
        self.assertEqual(plan.result_scope, "all")
        self.assertEqual(plan.tasks, ("literal_assignments",))
        validation = validate_plan_answer(
            plan,
            "PDCBVC literal assignments: 2 matching item(s).\n"
            "- line 1 TOP: A = '1'\nShowing the first 1 of 2 matching assignments.",
        )
        self.assertFalse(validation.passed)
        self.assertIn("exhaustive_result_truncated", validation.reasons)

    def test_more_followup_inherits_technical_plan_and_requests_all_results(self) -> None:
        state = SessionState(
            current_program="PDCBVC",
            current_intent="static_values",
            current_plan={
                "intent": "static_values",
                "operations": ["list"],
                "result_scope": "default",
            },
        )
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), intent="general",
            program_source="session",
        )
        plan = build_query_plan("there is more i think", scope, state=state)
        self.assertTrue(plan.explicit_followup)
        self.assertEqual(plan.intent, "static_values")
        self.assertEqual(plan.tasks, ("literal_assignments",))
        self.assertEqual(plan.result_scope, "all")
        conversational = QueryRoutingDecision(
            "conversational", "There is more?", category="conversational",
        )
        self.assertTrue(_routing_conflicts_with_verified_scope(conversational, plan))
        unclear = QueryRoutingDecision(
            "unclear", "Please provide more context.", category="clarification",
        )
        self.assertTrue(_routing_conflicts_with_verified_scope(unclear, plan))

    def test_semantic_merge_preserves_explicit_quality_tasks(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="dead_code")
        plan = build_query_plan(
            "Separate commented-out code, unreachable paragraphs, copybooks proven unused, "
            "and copybooks that only require review.",
            scope,
            intent="dead_code",
        )
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "dead_code",
            "tasks": ["unused_copybooks", "review_copybooks"],
            "relations": ["separate_categories"],
        })
        self.assertEqual(set(merged.tasks), {
            "commented_code", "unreachable_code", "unused_copybooks", "review_copybooks",
        })
        self.assertEqual(merged.relations, ("separate_categories",))

    def test_control_flow_word_after_does_not_become_call_context_relation(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="control_flow")
        plan = build_query_plan(
            "Walk through what happens after PDCBVC starts until it hands control away or ends.",
            scope,
            intent="control_flow",
        )
        self.assertEqual(plan.tasks, ("complete_program_flow",))
        self.assertNotIn("after", plan.relations)

    def test_production_language_triggers_variable_writes_without_lifecycle_keywords(self) -> None:
        # A variable can be "computed"/"calculated"/"produced" rather than "written" or
        # "set". The deterministic phase-detection must not depend on the LLM router
        # catching this paraphrase every time; it must reliably request the producer
        # claim (variable_writes) alongside the consumer claim (variable_reads).
        entity = EntityReference("PDCBVC", "variable", "WWORKFIELD", "PDCBVC|VARIABLE|WWORKFIELD")
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            intent="variable_dataflow",
        )
        plan = build_query_plan(
            "How is WWORKFIELD calculated in PDCBVC, and in which paragraphs is it later checked?",
            scope,
            intent="variable_dataflow",
        )
        self.assertIn("variable_writes", plan.tasks)
        self.assertIn("variable_reads", plan.tasks)
        capabilities = {subtask.capability for subtask in plan.subtasks}
        self.assertIn("variable_access", capabilities)

    def test_call_consequence_language_without_before_after_triggers_call_context(self) -> None:
        # Natural phrasings of "what happens after this call" do not always contain the
        # literal word "after". The deterministic layer must still recognize a
        # post-call/consequence question and request call_context, not a plain listing.
        entity = EntityReference("PDCBVC", "call", "PXRSEMAF", "PDCBVC|PXRSEMAF|CALL")
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            intent="external_programs",
        )
        plan = build_query_plan(
            "Once PDCBVC gets control back from the PXRSEMAF call, "
            "what does it check to decide what happens next?",
            scope,
            intent="external_programs",
        )
        self.assertEqual(plan.tasks, ("call_context",))
        self.assertIn("after", plan.relations)

    def test_call_preparation_language_without_before_triggers_call_context(self) -> None:
        entity = EntityReference("PDCBVC", "call", "PD1FS00", "PDCBVC|PD1FS00|CALL")
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            intent="external_programs",
        )
        plan = build_query_plan(
            "What does PDCBVC set up in preparation for calling PD1FS00?",
            scope,
            intent="external_programs",
        )
        self.assertEqual(plan.tasks, ("call_context",))
        self.assertIn("before", plan.relations)

    def test_whole_program_variable_question_routes_to_the_variable_catalogue(self) -> None:
        # No exact variable is resolvable, so the entity-scoped capabilities cannot
        # answer. The router must be able to select the program-wide catalogue as a
        # real destination instead of fabricating empty-entity claims.
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="variable_inventory")
        plan = build_query_plan(
            "name 10 variables inside PDCBVC", scope, intent="variable_inventory",
        )
        self.assertEqual(plan.intent, "variable_inventory")
        self.assertEqual(plan.tasks, ("variable_inventory",))
        self.assertEqual(plan.source_domains, ("dataflow.used_variables",))
        self.assertEqual([item.capability for item in plan.subtasks], ["variable_inventory"])
        self.assertEqual(plan.subtasks[0].entity_values, ())
        self.assertTrue(_direct_handler_supports(plan))
        self.assertIn("dataflow.used_variables", _chunk_types_for_plan(plan) or [])

    def test_variable_catalogue_intent_survives_semantic_merge(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="variable_inventory")
        plan = build_query_plan(
            "which data items does PDCBVC work with?", scope, intent="variable_inventory",
        )
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "variable_inventory",
            "domain": "dataflow",
            "tasks": ["variable_inventory"],
            "subtasks": [{
                "capability": "variable_inventory",
                "tasks": ["variable_inventory"],
                "entity_values": [],
            }],
        })
        self.assertEqual(merged.intent, "variable_inventory")
        self.assertEqual([item.capability for item in merged.subtasks], ["variable_inventory"])

    def test_named_variable_still_wins_over_the_whole_program_catalogue(self) -> None:
        # The catalogue must never displace exact entity evidence when the user
        # named a specific variable.
        entity = EntityReference("PDCBVC", "variable", "NPAGT", "PDCBVC|VARIABLE|NPAGT")
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            intent="variable_dataflow",
        )
        plan = build_query_plan(
            "Where is NPAGT written and later tested?", scope, intent="variable_inventory",
        )
        self.assertEqual(plan.intent, "variable_dataflow")
        self.assertNotIn("variable_inventory", plan.tasks)
        capabilities = {item.capability for item in plan.subtasks}
        self.assertIn("variable_access", capabilities)
        self.assertNotIn("variable_inventory", capabilities)

    def test_repeated_whole_program_claims_collapse_into_one_execution(self) -> None:
        # Several no-entity claims against one capability are one claim about the
        # same artifact family. Merging keeps every task but removes the duplicate
        # generation pass that made simple questions slow.
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="dead_code", domain="quality",
            tasks=("commented_code", "unreachable_code"),
            source_domains=("quality.dead_code", "program.comments"),
        )
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "dead_code",
            "tasks": ["commented_code", "unreachable_code"],
            "subtasks": [
                {"capability": "quality_evidence", "tasks": ["commented_code"], "entity_values": []},
                {"capability": "quality_evidence", "tasks": ["unreachable_code"], "entity_values": []},
            ],
        })
        quality_claims = [item for item in merged.subtasks if item.capability == "quality_evidence"]
        self.assertEqual(len(quality_claims), 1)
        self.assertEqual(set(quality_claims[0].tasks), {"commented_code", "unreachable_code"})

    def test_distinct_capabilities_are_not_collapsed_by_the_merge_guardrail(self) -> None:
        # Genuinely separate whole-program requests must stay independent claims.
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="datasets_tables",
            domain="integration", tasks=("db2_tables", "jcl_datasets"),
            source_domains=("architecture.db2_table", "jcl.file_io"),
        )
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "datasets_tables",
            "tasks": ["db2_tables", "jcl_datasets"],
            "relations": ["separate_categories"],
            "subtasks": [
                {"capability": "db2_evidence", "tasks": ["db2_tables"], "entity_values": []},
                {"capability": "jcl_evidence", "tasks": ["jcl_datasets"], "entity_values": []},
            ],
        })
        capabilities = [item.capability for item in merged.subtasks]
        self.assertIn("db2_evidence", capabilities)
        self.assertIn("jcl_evidence", capabilities)

    def test_paragraph_tasks_normalize_incompatible_xctl_intent(self) -> None:
        decision = _parse_routing_decision(json.dumps({
            "route": "technical", "intent": "external_programs", "domain": "integration",
            "tasks": ["paragraph_references", "paragraph_body"],
            "relations": ["referenced_by", "contains"],
            "source_domains": ["architecture.call_parameters"], "reply": "",
        }))
        self.assertEqual(decision.intent, "control_flow")
        self.assertEqual(decision.domain, "control_flow")
        self.assertEqual(decision.source_domains, ("controlflow.cfg",))

    def test_structured_paragraph_executor_separates_incoming_and_outgoing_edges(self) -> None:
        cfg = RetrievalResult(0.9, "\n".join([
            "edges[1].condition: EIBAID = DFHPF4",
            "edges[1].evidence: GO TO XCTL-LIV4.",
            "edges[1].from: BROWSE-FASE2",
            "edges[1].to: XCTL-LIV4",
            "edges[1].type: JUMP",
            "edges[2].evidence: PERFORM RESET-TWA THRU RESET-TWA-EXIT.",
            "edges[2].from: XCTL-LIV4",
            "edges[2].to: RESET-TWA",
            "edges[2].type: CALL_RANGE",
            "edges[3].evidence: GO TO XCTL-MAIN.",
            "edges[3].from: XCTL-LIV4",
            "edges[3].to: XCTL-MAIN",
            "edges[3].type: JUMP",
        ]), {"chunk_type": "controlflow.cfg"})
        context = RetrievalResult(
            0.8,
            "cobol-rekt paragraph logic: XCTL-LIV4 - MOVE '1' TO TWCOB-FASE - GO TO XCTL-MAIN; reads PROG-LIV4.",
            {"chunk_type": "integration.paragraph_context", "paragraph": "XCTL-LIV4"},
        )
        plan = QueryPlan(
            intent="control_flow", tasks=("paragraph_references", "paragraph_body"),
            relations=("referenced_by", "contains"),
            entities=(EntityReference(
                "PDCBVC", "paragraph", "XCTL-LIV4", "PDCBVC|PARAGRAPH|XCTL-LIV4"
            ),),
        )
        answer = _try_structured_plan_answer(plan, [cfg, context])
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("BROWSE-FASE2 when EIBAID = DFHPF4", answer)
        self.assertIn("CALL_RANGE to RESET-TWA", answer)
        self.assertIn("JUMP to XCTL-MAIN", answer)
        self.assertNotIn("XCTL-LIV4: `GO TO XCTL-MAIN`", answer)

    def test_structured_path_executor_traces_branches_to_terminal_transfers(self) -> None:
        cfg = RetrievalResult(0.9, "\n".join([
            "edges[1].from: START-PARA",
            "edges[1].to: CHECK-PARA",
            "edges[1].type: FALLTHROUGH",
            "edges[2].from: CHECK-PARA",
            "edges[2].to: RETURN-MAIN",
            "edges[2].type: JUMP",
            "edges[2].condition: FLAG = 'N'",
            "edges[2].evidence: GO TO RETURN-MAIN.",
            "edges[3].from: CHECK-PARA",
            "edges[3].to: XCTL-LIV5",
            "edges[3].type: JUMP",
            "edges[3].condition: FLAG = 'Y'",
            "edges[3].evidence: GO TO XCTL-LIV5.",
            "edges[4].from: XCTL-LIV5",
            "edges[4].to: RESET-TWA",
            "edges[4].type: CALL_RANGE",
            "edges[5].from: XCTL-LIV5",
            "edges[5].to: XCTL-MAIN",
            "edges[5].type: JUMP",
            "edges[5].evidence: GO TO XCTL-MAIN.",
        ]), {"chunk_type": "controlflow.cfg"})
        cics = RetrievalResult(0.8, "\n".join([
            "content.operations[0].command: XCTL",
            "content.operations[0].line_start: 845",
            "content.operations[0].paragraph: XCTL-MAIN",
            "content.operations[0].statement: EXEC CICS XCTL PROGRAM('PDPRED') END-EXEC.",
            "content.operations[1].command: RETURN",
            "content.operations[1].line_start: 846",
            "content.operations[1].paragraph: XCTL-MAIN",
            "content.operations[1].statement: EXEC CICS RETURN END-EXEC.",
        ]), {"chunk_type": "architecture.cics_operations"})
        plan = QueryPlan(
            intent="control_flow", tasks=("path_from_paragraph",), relations=("starts_at",),
            entities=(EntityReference(
                "PDCBVC", "paragraph", "START-PARA", "PDCBVC|PARAGRAPH|START-PARA"
            ),),
        )
        answer = _try_structured_plan_answer(plan, [cfg, cics])
        assert answer is not None
        self.assertIn("START-PARA -> CHECK-PARA", answer)
        self.assertIn("CHECK-PARA -> RETURN-MAIN", answer)
        self.assertIn("CHECK-PARA -> XCTL-LIV5", answer)
        self.assertIn("XCTL-LIV5 -> XCTL-MAIN", answer)
        self.assertIn("EXEC CICS XCTL PROGRAM('PDPRED')", answer)
        self.assertIn("source line 845", answer)
        self.assertIn("EXEC CICS RETURN END-EXEC", answer)
        self.assertIn("source line 846", answer)
        self.assertNotIn("RESET-TWA", answer)

    def test_structured_dataset_executor_keeps_db2_and_jcl_separate(self) -> None:
        sources = [
            RetrievalResult(0.9, "content.stmt_type: selectIntoStatement\ncontent.table: DUAL", {"chunk_type": "architecture.db2_table", "db2_table": "DUAL"}),
            RetrievalResult(0.8, "content.has_jcl_linkage: false\ncontent.matching_jobs_count: 0", {"chunk_type": "jcl.file_io"}),
        ]
        plan = QueryPlan(intent="datasets_tables", tasks=("db2_tables", "jcl_datasets"), relations=("separate_categories",))
        answer = _try_structured_plan_answer(plan, sources)
        assert answer is not None
        self.assertIn("DB2 tables:", answer)
        self.assertIn("DUAL: selectIntoStatement", answer)
        self.assertIn("JCL datasets:", answer)
        self.assertIn("No program-to-JCL linkage", answer)

    def test_structured_quality_executor_distinguishes_unused_from_review(self) -> None:
        quality = RetrievalResult(
            0.9,
            "content.cfg_reachability.unreachable_nodes_count: 0\n"
            "content.commented_out_code[0].line: 433\n"
            "content.commented_out_code[0].paragraph: INIZ-PARAM\n"
            "content.commented_out_code[0].text: MOVE '20' TO FIELD.",
            {"chunk_type": "quality.dead_code"},
        )
        copybooks = RetrievalResult(
            0.8,
            "content.copybook_status[0].copybook: DFHBMSCA\n"
            "content.copybook_status[0].status: needs_review_no_reference_in_available_artifacts\n"
            "content.needs_review_copybooks: DFHBMSCA, PDCBVCM, PDRTIP01, PDRVC",
            {"chunk_type": "architecture.unused_copybooks"},
        )
        plan = QueryPlan(intent="dead_code", tasks=("commented_code", "unreachable_code", "unused_copybooks", "review_copybooks"), relations=("separate_categories",))
        answer = _try_structured_plan_answer(plan, [quality, copybooks])
        assert answer is not None
        self.assertIn("line 433 in INIZ-PARAM", answer)
        self.assertIn("all CFG nodes are reachable", answer)
        self.assertIn("None are proven unused", answer)
        self.assertIn("DFHBMSCA", answer)
        self.assertIn("PDCBVCM", answer)
        self.assertIn("PDRTIP01", answer)
        self.assertIn("PDRVC", answer)

    def test_multisource_retrieval_balances_results_per_program(self) -> None:
        def outcome(program: str) -> RetrievalOutcome:
            item = RetrievalResult(0.9, f"Evidence for {program}", {"source_id": program, "program": program})
            return RetrievalOutcome(
                results=[item], vector_results=[item], lexical_results=[], filters={"program": program},
                intent="control_flow", correction_applied=False, expanded_count=0,
                guard=EvidenceGuard("sufficient"),
            )
        plan = QueryPlan(
            intent="control_flow", program="PROGA", programs=("PROGA", "PROGB"),
            category="multi_source_comparison", requires_comparison=True,
        )
        scope = QueryScope(program="PROGA", programs=("PROGA", "PROGB"), intent="control_flow")
        with patch("cobol_rag.query.retrieve_with_trace", side_effect=[outcome("PROGA"), outcome("PROGB")]) as retrieval:
            result = _retrieve_for_plan(
                "Compare PROGA and PROGB", config=AppConfig(), plan=plan, scope=scope,
                top_k=None, chunk_types=None,
            )
        self.assertEqual(retrieval.call_count, 2)
        self.assertEqual([call.kwargs["program"] for call in retrieval.call_args_list], ["PROGA", "PROGB"])
        self.assertEqual({item.metadata["program"] for item in result.results}, {"PROGA", "PROGB"})
        self.assertEqual(result.filters["strategy"], "balanced_per_program")
        self.assertEqual(result.guard.status, "sufficient")

    def test_semantic_route_descriptions_recover_general_llm_intent(self) -> None:
        cases = {
            "Describe the decisions PDCBVC makes and their resulting actions using direct code evidence.": "business_rules",
            "Walk through what happens after PDCBVC starts until it hands control away or ends.": "control_flow",
            "Show the transaction-processing instructions issued by PDCBVC and their code locations.": "cics_operations",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                hint = _semantic_route_hint(question)
                self.assertIsNotNone(hint)
                assert hint is not None
                self.assertEqual(hint[0], expected)
        self.assertIsNone(_semantic_route_hint("what is the weather in Rome now"))

    def test_llm_semantic_plan_is_not_overridden_by_keyword_similarity(self) -> None:
        response = type("Response", (), {"text": json.dumps({
            "route": "technical", "intent": "variable_dataflow",
            "category": "single_source", "domain": "dataflow",
            "tasks": ["variable_definition"], "relations": [],
            "operations": ["describe"], "source_domains": ["dataflow.variable"],
            "confidence": 0.45, "reply": "",
        })})()
        question = "Describe the decisions PDCBVC makes and their resulting actions using direct code evidence."
        with patch("cobol_rag.query.build_llm") as build:
            build.return_value.complete.return_value = response
            decision = _route_query(
                question, AppConfig(),
                preliminary_plan=QueryPlan(intent="general", program="PDCBVC", programs=("PDCBVC",)),
                preliminary_scope=QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general"),
            )
        self.assertEqual(decision.intent, "variable_dataflow")
        self.assertEqual(decision.planner_source, "semantic_llm")
        self.assertEqual(decision.tasks, ("variable_definition",))

    def test_unclear_llm_decision_is_not_rewritten_by_keyword_similarity(self) -> None:
        response = type("Response", (), {"text": json.dumps({
            "route": "unclear", "intent": "general", "domain": "general",
            "tasks": [], "relations": [], "category": "clarification", "confidence": 0.99,
            "reply": "Please clarify.",
        })})()
        question = "Walk through what happens after PDCBVC starts until it hands control away or ends."
        with patch("cobol_rag.query.build_llm") as build:
            build.return_value.complete.return_value = response
            decision = _route_query(
                question, AppConfig(),
                preliminary_plan=QueryPlan(intent="general", program="PDCBVC", programs=("PDCBVC",)),
                preliminary_scope=QueryScope(
                    program="PDCBVC", programs=("PDCBVC",), intent="general",
                    program_source="question",
                ),
            )
        self.assertEqual(decision.route, "unclear")
        self.assertEqual(decision.intent, "general")
        self.assertEqual(decision.planner_source, "semantic_llm")

    def test_conversational_route_cannot_answer_a_program_scoped_question(self) -> None:
        # The conversational route bypasses retrieval and citation validation. A
        # message that names an analyzed program must never be answered that way,
        # otherwise the model narrates COBOL facts that were never retrieved.
        conversational = type("Response", (), {"text": json.dumps({
            "route": "conversational", "category": "conversational",
            "domain": "conversation", "intent": "general", "tasks": [], "relations": [],
            "response_language": "en", "confidence": 0.9,
            "reply": "Here are 10 variables inside PDCBVC: variable1, variable2, variable3.",
        })})()
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), intent="general",
            program_source="question",
        )
        plan = QueryPlan(intent="general", program="PDCBVC", programs=("PDCBVC",))
        with patch("cobol_rag.query.build_llm") as build:
            build.return_value.complete.return_value = conversational
            decision = _route_query(
                "name 10 variables inside PDCBVC", AppConfig(),
                preliminary_plan=plan, preliminary_scope=scope,
            )
        self.assertEqual(decision.route, "technical")
        self.assertNotIn("variable1", decision.reply)

    def test_routing_shortlist_keeps_room_for_a_second_capability(self) -> None:
        # A compound request needs one capability per claim. Narrowing to the single
        # best match would make the second claim unrepresentable, so the shortlist
        # must stay wider than the top hit.
        ranked = tuple(
            CapabilityMatch(name, score, 0.05)
            for name, score in (
                ("db2_evidence", 0.71), ("jcl_evidence", 0.66), ("copybook_evidence", 0.52),
                ("program_summary", 0.47), ("control_flow", 0.30),
            )
        )
        with patch("cobol_rag.query._rank_question_capabilities", return_value=ranked):
            candidates = _routing_candidates(
                "List the DB2 tables and the JCL datasets separately.",
                AppConfig(), QueryScope(program="PDCBVC", programs=("PDCBVC",)),
            )
        names = [match.capability for match in candidates]
        self.assertIn("db2_evidence", names)
        self.assertIn("jcl_evidence", names)
        self.assertNotIn("control_flow", names)

    def test_shortlist_always_keeps_the_capability_the_deterministic_layer_chose(self) -> None:
        # Entity-scoped questions score low against prose descriptions because the
        # identifier matches no wording, yet those are exactly the ones the
        # deterministic layer already resolved. Constraining the planner must never
        # remove a conclusion that was already reached.
        ranked = tuple(
            CapabilityMatch(name, score, 0.02)
            for name, score in (
                ("quality_evidence", 0.51), ("program_summary", 0.49), ("control_flow", 0.47),
            )
        )
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="external_programs",
            domain="integration", tasks=("call_context",),
        )
        with patch("cobol_rag.query._rank_question_capabilities", return_value=ranked):
            candidates = _routing_candidates(
                "What does PDCBVC set up in preparation for calling PD1FS00?",
                AppConfig(), QueryScope(program="PDCBVC", programs=("PDCBVC",)), plan,
            )
        self.assertIn("call_context", [match.capability for match in candidates])

    def test_shortlist_is_empty_only_when_ranking_returns_nothing(self) -> None:
        with patch("cobol_rag.query._rank_question_capabilities", return_value=()):
            self.assertEqual(_routing_candidates("hi", AppConfig(), QueryScope()), ())

    def test_plan_outside_the_shortlist_is_rejected(self) -> None:
        candidates = (
            CapabilityMatch("call_context", 0.67, 0.06),
            CapabilityMatch("call_evidence", 0.61, 0.0),
        )
        inside = QueryRoutingDecision(
            "technical", "", "external_programs", tasks=("call_context",),
            subtasks=({"capability": "call_context", "tasks": ["call_context"]},),
        )
        outside = QueryRoutingDecision(
            "technical", "", "program_summary", tasks=("program_summary",),
            subtasks=({"capability": "program_summary", "tasks": ["program_summary"]},),
        )
        self.assertTrue(_decision_respects_candidates(inside, candidates))
        self.assertFalse(_decision_respects_candidates(outside, candidates))

    def test_conversational_plans_are_never_constrained_by_the_shortlist(self) -> None:
        candidates = (CapabilityMatch("program_summary", 0.60, 0.05),)
        chat = QueryRoutingDecision("conversational", "Hello!", category="conversational")
        self.assertTrue(_decision_respects_candidates(chat, candidates))

    def test_constrained_prompt_lists_only_the_shortlisted_capabilities(self) -> None:
        candidates = (
            CapabilityMatch("call_context", 0.67, 0.06),
            CapabilityMatch("call_evidence", 0.61, 0.0),
        )
        prompt = _build_constrained_routing_prompt(
            "What does PDCBVC set up before calling PD1FS00?",
            None, None,
            preliminary_plan=QueryPlan(program="PDCBVC", programs=("PDCBVC",)),
            preliminary_scope=QueryScope(program="PDCBVC", programs=("PDCBVC",)),
            candidates=candidates,
        )
        self.assertIn("call_context", prompt)
        self.assertIn("call_evidence", prompt)
        self.assertNotIn("pagination_evidence", prompt)
        self.assertNotIn("copybook_evidence", prompt)
        # The point of constraining is a smaller decision than the full planner.
        full = _build_routing_prompt(
            "What does PDCBVC set up before calling PD1FS00?", None, None,
            preliminary_plan=QueryPlan(program="PDCBVC", programs=("PDCBVC",)),
            preliminary_scope=QueryScope(program="PDCBVC", programs=("PDCBVC",)),
        )
        self.assertLess(len(prompt), len(full) / 2)

    def test_planner_cannot_invent_an_exclusion_the_user_never_asked_for(self) -> None:
        # Observed production failure: "What external programs does PDCBVC call?"
        # returned 4 of 5 calls because the planner added excluded_operations XCTL,
        # silently dropping PDPRED. An exclusion removes real evidence, so it may
        # only come from the user's own wording.
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="external_programs")
        plan = build_query_plan(
            "What external programs does PDCBVC call, and what parameters go with each?",
            scope, intent="external_programs",
        )
        self.assertEqual(plan.excluded_operations, ())
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "external_programs",
            "tasks": ["external_calls"],
            "excluded_operations": ["XCTL"],
        })
        self.assertEqual(merged.excluded_operations, ())

    def test_explicit_user_exclusion_is_still_honoured(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="external_programs")
        plan = build_query_plan(
            "Show only LINK calls in PDCBVC; exclude XCTL.", scope, intent="external_programs",
        )
        merged = merge_semantic_plan(plan, {"route": "technical", "intent": "external_programs"})
        self.assertEqual(merged.excluded_operations, ("XCTL",))

    def test_lines_of_code_is_a_size_question_not_a_source_line_request(self) -> None:
        # Observed production failure: "in terms of lines of code" set the
        # source_line output field, and the contract then rejected a correct
        # program summary for omitting a line number it never renders.
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="program_summary")
        plan = build_query_plan(
            "How big is PDCBVC in terms of lines of code?", scope, intent="program_summary",
        )
        self.assertNotIn("source_line", plan.output_fields)

    def test_program_level_answers_are_not_required_to_cite_a_source_line(self) -> None:
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="program_summary",
            domain="program_structure", tasks=("program_summary",),
            output_fields=("source_line",),
        )
        validation = validate_plan_answer(
            plan, "PDCBVC is a COBOL CICS program with approximately 392 LOC.",
        )
        self.assertTrue(validation.passed, validation.reasons)

    def test_entity_answers_still_require_the_requested_source_line(self) -> None:
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="variable_dataflow",
            domain="dataflow", tasks=("variable_writes",), output_fields=("source_line",),
        )
        validation = validate_plan_answer(plan, "NPAGT is modified in CALCOLA-NPAG.")
        self.assertFalse(validation.passed)
        self.assertIn("missing_requested_field:source_line", validation.reasons)

    def test_capability_ranking_orders_by_similarity_and_reports_the_margin(self) -> None:
        question = [1.0, 0.0, 0.0]
        descriptors = {
            "call_evidence": [0.0, 1.0, 0.0],
            "variable_inventory": [1.0, 0.0, 0.0],
            "db2_evidence": [0.6, 0.8, 0.0],
        }
        ranked = rank_capabilities(question, descriptors)
        self.assertEqual(ranked[0].capability, "variable_inventory")
        self.assertAlmostEqual(ranked[0].score, 1.0, places=6)
        self.assertGreater(ranked[0].margin, 0.0)
        self.assertTrue(ranked[0].confident)

    def test_entity_scope_gates_which_capabilities_can_be_ranked(self) -> None:
        # Without a resolved identifier the entity-scoped capabilities have nothing
        # to answer about, and a named variable must not be answered by a catalogue.
        program_wide = eligible_capabilities(entity_types=())
        self.assertIn("variable_inventory", program_wide)
        self.assertNotIn("variable_access", program_wide)
        self.assertNotIn("call_context", program_wide)

        named_variable = eligible_capabilities(entity_types=("variable",))
        self.assertIn("variable_access", named_variable)
        self.assertNotIn("variable_inventory", named_variable)

    def test_weak_technical_route_receives_a_capability_from_semantic_ranking(self) -> None:
        # The planner returned "technical" with no intent and no task, which selects
        # no evidence handler at all. Semantic ranking supplies the missing one.
        empty = QueryRoutingDecision("technical", "", "general")
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = QueryPlan(intent="general", program="PDCBVC", programs=("PDCBVC",))
        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("variable_inventory", 0.71, 0.09),),
        ):
            resolved = _resolve_weak_technical_route(
                "name a few fields in PDCBVC", AppConfig(), scope, plan, empty,
            )
        self.assertEqual(resolved.intent, "variable_inventory")
        self.assertEqual(resolved.tasks, ("variable_inventory",))
        self.assertEqual(resolved.planner_source, "capability_router")

    def test_low_confidence_ranking_does_not_override_the_planner(self) -> None:
        empty = QueryRoutingDecision("technical", "", "general")
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = QueryPlan(intent="general", program="PDCBVC", programs=("PDCBVC",))
        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("variable_lineage", 0.46, 0.006),),
        ):
            resolved = _resolve_weak_technical_route(
                "something unrelated", AppConfig(), scope, plan, empty,
            )
        self.assertIs(resolved, empty)

    def test_capability_ranking_never_overrides_a_resolved_planner_route(self) -> None:
        strong = QueryRoutingDecision(
            "technical", "", "external_programs", tasks=("external_calls",),
        )
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="external_programs")
        plan = QueryPlan(intent="external_programs", program="PDCBVC", programs=("PDCBVC",))
        with patch("cobol_rag.query._rank_question_capabilities") as ranker:
            resolved = _resolve_weak_technical_route(
                "which programs are called", AppConfig(), scope, plan, strong,
            )
        ranker.assert_not_called()
        self.assertIs(resolved, strong)

    def test_technical_question_without_identifiers_cannot_be_answered_conversationally(self) -> None:
        # Observed production failure: "Explain how paging through results works in
        # this program" was answered conversationally with invented prose. It names
        # no program and no identifier, so only semantic ranking can catch it.
        chat = QueryRoutingDecision(
            "conversational",
            "In this program, paging through results is handled by a pagination logic.",
            category="conversational",
        )
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("pagination_evidence", 0.87, 0.20),),
        ):
            blocked = _conversational_route_is_blocked(
                "Explain how paging through results works in this program.",
                AppConfig(), chat, scope,
            )
        self.assertTrue(blocked)

    def test_small_talk_is_not_blocked_by_capability_ranking(self) -> None:
        chat = QueryRoutingDecision(
            "conversational", "Hello! How can I help?", category="conversational",
        )
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("program_summary", 0.45, 0.015),),
        ):
            blocked = _conversational_route_is_blocked("hi", AppConfig(), chat, scope)
        self.assertFalse(blocked)

    def test_conversational_reply_stating_a_cobol_fact_is_rejected(self) -> None:
        # Observed production failure: the follow-up "only?" produced a fluent
        # conversational description of the program's business purpose with no
        # evidence at all. The conversational route runs no validation, so a reply
        # that names the program is a technical answer that skipped every check.
        invented = QueryRoutingDecision(
            "conversational",
            "PDCBVC is a mainframe program that performs data conversion and "
            "business validation for a banking system.",
            category="conversational",
        )
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), intent="general",
            program_source="session",
        )
        plan = QueryPlan(intent="general", program="PDCBVC", programs=("PDCBVC",))
        self.assertTrue(_routing_conflicts_with_verified_scope(invented, plan, scope))

    def test_conversational_capability_reply_is_still_allowed(self) -> None:
        # Naming the domain is not a claim about analyzed code.
        safe = QueryRoutingDecision(
            "conversational",
            "I can answer questions about COBOL programs, their calls and variables.",
            category="conversational",
        )
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), intent="general",
            program_source="target",
        )
        plan = QueryPlan(intent="general", program="PDCBVC", programs=("PDCBVC",))
        self.assertFalse(_routing_conflicts_with_verified_scope(safe, plan, scope))

    def test_explicit_technical_followup_cannot_be_answered_conversationally(self) -> None:
        # Observed production failure: "there is more, show me the rest" returned
        # "Sure, here's the rest of the program summary: ..." with no evidence.
        continuation = QueryRoutingDecision(
            "conversational", "Sure, here's the rest: ...", category="conversational",
        )
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), intent="general",
            program_source="session",
        )
        plan = QueryPlan(
            intent="general", program="PDCBVC", programs=("PDCBVC",),
            explicit_followup=True,
        )
        self.assertTrue(_routing_conflicts_with_verified_scope(continuation, plan, scope))

    def test_greeting_with_a_selected_program_stays_conversational(self) -> None:
        # A UI-selected program must not turn small talk into a technical turn.
        conversational = type("Response", (), {"text": json.dumps({
            "route": "conversational", "category": "conversational",
            "domain": "conversation", "intent": "general", "tasks": [], "relations": [],
            "response_language": "en", "confidence": 0.9, "reply": "Hello! How can I help?",
        })})()
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), intent="general",
            program_source="target",
        )
        with patch("cobol_rag.query.build_llm") as build:
            build.return_value.complete.return_value = conversational
            decision = _route_query(
                "hi", AppConfig(),
                preliminary_plan=QueryPlan(intent="general", program="PDCBVC", programs=("PDCBVC",)),
                preliminary_scope=scope,
            )
        self.assertEqual(decision.route, "conversational")
        self.assertEqual(decision.reply, "Hello! How can I help?")

    def test_pagination_literal_constraint_builds_a_technical_task(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = build_query_plan(
            "Explain the pagination logic in PDCBVC using direct evidence.", scope, intent="general",
        )
        self.assertEqual(plan.intent, "control_flow")
        self.assertEqual(plan.tasks, ("pagination_logic",))

    def test_conversational_llm_routes_cannot_override_verified_technical_plan(self) -> None:
        conversational = type("Response", (), {"text": json.dumps({
            "route": "conversational", "intent": "pagination_logic",
            "domain": "control_flow", "tasks": [], "relations": [],
            "response_language": "en", "reply": "Please provide more details.",
        })})()
        preliminary = QueryPlan(
            intent="control_flow", domain="control_flow", tasks=("pagination_logic",),
            operations=("describe",), program="PDCBVC", programs=("PDCBVC",),
        )
        with patch("cobol_rag.query.build_llm") as build:
            build.return_value.complete.side_effect = [conversational, conversational]
            decision = _route_query(
                "Explain the pagination logic in PDCBVC using direct evidence.", AppConfig(),
                preliminary_plan=preliminary,
                preliminary_scope=QueryScope(
                    program="PDCBVC", programs=("PDCBVC",), intent="control_flow",
                    program_source="question",
                ),
            )
        self.assertEqual(decision.route, "technical")
        self.assertEqual(decision.intent, "control_flow")
        self.assertEqual(decision.tasks, ("pagination_logic",))
        self.assertEqual(decision.planner_source, "deterministic_fallback")

    def test_invalid_full_plan_retries_with_compact_llm_router(self) -> None:
        invalid = type("Response", (), {"text": "not-json"})()
        recovered = type("Response", (), {"text": json.dumps({
            "route": "conversational", "category": "conversational",
            "domain": "conversation", "intent": "general", "tasks": [],
            "relations": [], "response_language": "it", "reply": "Sì, posso rispondere in italiano.",
        })})()
        with patch("cobol_rag.query.build_llm") as build:
            build.return_value.complete.side_effect = [invalid, recovered]
            decision = _route_query("Can you answer in Italian?", AppConfig())
        self.assertEqual(build.call_count, 2)
        self.assertEqual(decision.route, "conversational")
        self.assertEqual(decision.response_language, "it")
        self.assertEqual(decision.tasks, ())

    def test_llm_routes_unrelated_question_out_of_scope(self) -> None:
        response = type("Response", (), {"text": json.dumps({
            "route": "unclear", "intent": "general",
            "category": "out_of_scope", "confidence": 0.99,
            "reply": "I can only answer questions about the analyzed COBOL evidence.",
        })})()
        with patch("cobol_rag.query.build_llm") as build:
            build.return_value.complete.return_value = response
            decision = _route_query("what is the weather in Rome now", AppConfig())
        self.assertEqual(decision.route, "unclear")
        self.assertEqual(decision.category, "out_of_scope")
        self.assertIn("COBOL", decision.reply)

    def test_router_normalizes_clarification_reply_to_unclear(self) -> None:
        decision = _parse_routing_decision(
            '{"route":"conversational","intent":"general",'
            '"reply":"I do not understand; please provide more context."}'
        )
        self.assertEqual(decision.route, "unclear")

    def test_structured_entity_followup_routes_without_llm(self) -> None:
        state = SessionState(
            current_program="PDCBVC",
            current_entity_type="variable",
            current_entity_value="PD1VOCI-RETURN",
            current_entity_key="PDCBVC|VARIABLE|PD1VOCI-RETURN",
            current_intent="variable_dataflow",
        )
        decision = _deterministic_routing(
            "Where else is it checked?",
            QueryScope(
                program="PDCBVC",
                entity_type="variable",
                entity_value="PD1VOCI-RETURN",
                entity_key="PDCBVC|VARIABLE|PD1VOCI-RETURN",
                intent="variable_dataflow",
                entity_source="session",
            ),
            state,
        )
        self.assertEqual(decision, QueryRoutingDecision("technical", "", "variable_dataflow"))

    def test_conversational_turn_is_not_added_to_technical_history(self) -> None:
        session = ChatSession(config=AppConfig())
        session.turns.extend(
            [
                ChatTurn("What calls are made?", "Answer", [], route="technical"),
                ChatTurn("A social message", "Hello", [], route="conversational"),
            ]
        )
        history = session._history_context()
        self.assertIn("What calls are made?", history or "")
        self.assertNotIn("A social message", history or "")

    def test_history_does_not_reinject_assistant_claims(self) -> None:
        session = ChatSession(config=AppConfig())
        session.turns.append(
            ChatTurn(
                user="What are the rules?",
                assistant="Invented customer account claim",
                sources=[],
            )
        )
        history = session._history_context()
        self.assertIn("What are the rules?", history or "")
        self.assertNotIn("Invented customer account claim", history or "")

    def test_cancelled_question_leaves_no_trace_in_chat_memory(self) -> None:
        # The worker thread cannot be interrupted, so the answer still arrives.
        # What must not happen is the next question resolving against a turn the
        # user stopped waiting for.
        session = ChatSession(config=AppConfig())

        def answer_after_a_cancel(**kwargs):
            session.cancel()
            return QueryAnswer(
                question="Which paragraphs modify NPAGT?",
                answer="NPAGT direct COBOL access evidence: line 397.",
                sources=[],
                route="technical",
                scope=QueryScope(intent="variable_dataflow", program="PDCBVC"),
            )

        with patch("cobol_rag.chat.answer_query", side_effect=answer_after_a_cancel):
            result = session.ask("Which paragraphs modify NPAGT?")

        self.assertEqual(result.execution_mode, "cancelled")
        self.assertEqual(session.turns, [])
        self.assertIsNone(session._history_context())

    def test_uncancelled_question_is_recorded_as_usual(self) -> None:
        session = ChatSession(config=AppConfig())
        answer = QueryAnswer(
            question="Which paragraphs modify NPAGT?",
            answer="NPAGT direct COBOL access evidence: line 397.",
            sources=[],
            route="technical",
            scope=QueryScope(intent="variable_dataflow", program="PDCBVC"),
        )

        with patch("cobol_rag.chat.answer_query", return_value=answer):
            result = session.ask("Which paragraphs modify NPAGT?")

        self.assertNotEqual(result.execution_mode, "cancelled")
        self.assertEqual(len(session.turns), 1)
        self.assertIn("Which paragraphs modify NPAGT?", session._history_context() or "")

    def test_cancel_only_discards_the_question_that_was_running(self) -> None:
        # A cancel raises the generation once; the question asked afterwards
        # starts from the new generation and must be kept.
        session = ChatSession(config=AppConfig())
        session.cancel()
        answer = QueryAnswer(
            question="What gives NPAGT its value?",
            answer="Modified at CALCOLA-NPAG, line 397.",
            sources=[],
            route="technical",
            scope=QueryScope(intent="variable_dataflow", program="PDCBVC"),
        )

        with patch("cobol_rag.chat.answer_query", return_value=answer):
            result = session.ask("What gives NPAGT its value?")

        self.assertNotEqual(result.execution_mode, "cancelled")
        self.assertEqual(len(session.turns), 1)

    def test_prompt_respects_evidence_character_budget(self) -> None:
        sources = [
            RetrievalResult(0.9, "A" * 180, {"source_id": "one"}),
            RetrievalResult(0.8, "B" * 180, {"source_id": "two"}),
        ]
        prompt = _build_prompt(
            question="Question?",
            sources=sources,
            max_context_chars=240,
        )
        self.assertIn("source_id: one", prompt)
        self.assertNotIn("source_id: two", prompt)

    def test_missing_model_citations_are_added_from_returned_sources(self) -> None:
        sources = [
            RetrievalResult(
                0.9,
                "evidence",
                {"source_file": "business_rule/business_rule.BR-028.json"},
            )
        ]
        answer = _ensure_citations("Grounded answer.", sources)
        self.assertIn("[Source 1] business_rule/business_rule.BR-028.json", answer)


class EntityScopedEvidenceContractTest(unittest.TestCase):
    """Evidence offered as being about an identifier must mention that identifier."""

    @staticmethod
    def _plan(value: str, *, intent: str, tasks: tuple[str, ...]) -> QueryPlan:
        entity = EntityReference(
            "PDCBVC", "variable", value, f"PDCBVC|VARIABLE|{value}",
        )
        return QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent=intent,
            domain="control_flow", tasks=tasks, entities=(entity,),
        )

    def test_off_entity_rule_dump_is_rejected_under_any_intent(self) -> None:
        # A claim scoped to NPAGT that comes back with the whole business-rule
        # catalogue has answered a different question, and the intent it was
        # routed under must not excuse it.
        plan = self._plan("NPAGT", intent="business_rules", tasks=("business_rules",))
        answer = (
            "PDCBVC business rules (32 matching unique direct-evidence rule(s)):\n"
            "- BR-001\n  Condition: NOT (TWCOB-FASE = '1' OR TWCOB-FASE = '2')\n"
            "  Action: JUMP -> ABEND00\n  Source location: PDCBVC.CBL line 227\n"
            "- BR-010\n  Condition: EIBAID = DFHENTER\n"
            "  Action: JUMP -> BROWSE-FASE2-ENTER\n  Source location: PDCBVC.CBL line 299\n"
        )

        contract = validate_plan_answer(plan, answer)

        self.assertFalse(contract.passed)
        self.assertIn("missing_requested_entity:NPAGT", contract.reasons)

    def test_on_entity_answer_passes_under_the_same_intent(self) -> None:
        plan = self._plan("TWCOB-FASE", intent="business_rules", tasks=("business_rules",))
        answer = (
            "PDCBVC business rules (1 matching unique direct-evidence rule(s)):\n"
            "- BR-001\n  Condition: NOT (TWCOB-FASE = '1' OR TWCOB-FASE = '2')\n"
            "  Action: JUMP -> ABEND00\n  Source location: PDCBVC.CBL line 227\n"
        )

        self.assertTrue(validate_plan_answer(plan, answer).passed)

    def test_group_item_is_satisfied_by_a_qualified_child(self) -> None:
        # PD1VOCI is a group; evidence that names only its children is still
        # evidence about it, so the contract must not reject the record.
        plan = self._plan("PD1VOCI", intent="variable_dataflow", tasks=("variable_reads",))
        answer = (
            "PD1VOCI direct COBOL access evidence:\nTested/read at:\n"
            "- MUOVI-DATI-10, line 676: `IF PD1VOCI-IND GREATER PD1VOCI-TABVOX-NUMERO THEN`\n"
        )

        self.assertTrue(validate_plan_answer(plan, answer).passed)

    def test_specific_field_is_not_satisfied_by_the_group_alone(self) -> None:
        plan = self._plan(
            "PD1VOCI-TABVOX-NUMERO", intent="variable_dataflow", tasks=("variable_reads",),
        )
        answer = "PD1VOCI is passed as the COMMAREA on the LINK at line 486.\nRead sites: none recorded."

        contract = validate_plan_answer(plan, answer)

        self.assertFalse(contract.passed)
        self.assertIn("missing_requested_entity:PD1VOCI-TABVOX-NUMERO", contract.reasons)

    def test_reporting_absence_of_evidence_for_the_entity_passes(self) -> None:
        # Abstaining is a legitimate answer as long as it is about the entity
        # that was asked for, so it must not be rejected as off-entity.
        plan = self._plan("WCTPAG", intent="external_programs", tasks=("external_calls",))
        answer = "No outgoing call evidence matched WCTPAG in PDCBVC."

        self.assertTrue(validate_plan_answer(plan, answer).passed)

    def test_program_wide_claims_are_untouched_by_the_entity_contract(self) -> None:
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="program_summary",
            domain="program_structure", tasks=("program_summary",),
        )
        answer = "PDCBVC technical overview: 392 lines of code, 72 statements."

        contract = validate_plan_answer(plan, answer)

        self.assertTrue(contract.passed)
        self.assertFalse(
            [reason for reason in contract.reasons if reason.startswith("missing_requested_entity")]
        )


class AbsentCapabilityRoutingTest(unittest.TestCase):
    """A capability the analysis proved empty answers as empty, not by proxy."""

    MANIFEST = {
        "capabilities": {
            "jcl_evidence": {
                "available": False,
                "count": 0,
                "reason": "no program-to-JCL linkage in the parsed job steps",
                "artifact": "program.capability_manifest.json",
            },
            "program_summary": {"available": True, "count": 1},
            "cics_evidence": {"available": True, "count": 18},
        }
    }

    def test_missing_capability_is_offered_to_the_ranker_not_filtered_out(self) -> None:
        # Removing the missing capability before ranking is what sent "is there
        # any batch JCL" to the nearest capability that did have evidence.
        captured: dict[str, object] = {}

        def rank(question: str, *, allowed=None):
            captured["allowed"] = set(allowed)
            return (CapabilityMatch("jcl_evidence", 0.69, 0.21),)

        with (
            patch("cobol_rag.query.unavailable_capabilities", return_value=frozenset({"jcl_evidence"})),
            patch("cobol_rag.query.router_for", return_value=Mock(rank=rank)),
        ):
            capability = _absent_capability_request(
                "Is there any batch JCL associated with PDCBVC?",
                Mock(),
                QueryScope(intent="general", program="PDCBVC"),
            )

        self.assertEqual(capability, "jcl_evidence")
        self.assertIn("jcl_evidence", captured["allowed"])

    def test_absence_is_not_the_answer_when_present_evidence_covers_the_entity(self) -> None:
        # Asking how a variable reaches a screen field is answerable from its
        # dataflow even with no screen lineage produced. Reporting the missing
        # capability there refused a question the evidence could answer.
        scope = QueryScope(
            intent="general", program="PDCBVC",
            entities=(EntityReference("PDCBVC", "variable", "RIGA-MAPPA", "PDCBVC|VARIABLE|RIGA-MAPPA"),),
        )
        with (
            patch("cobol_rag.query.unavailable_capabilities", return_value=frozenset({"screen_lineage"})),
            patch(
                "cobol_rag.query.router_for",
                return_value=Mock(rank=lambda q, allowed=None: (CapabilityMatch("screen_lineage", 0.71, 0.12),)),
            ),
        ):
            self.assertIsNone(_absent_capability_request(
                "Explain how RIGA-MAPPA is built and which map field receives it.",
                Mock(),
                scope,
            ))

    def test_absence_still_answers_when_no_entity_can_be_served(self) -> None:
        # No identifier means no per-entity evidence to fall back on, so the
        # manifest's recorded absence remains the answer.
        with (
            patch("cobol_rag.query.unavailable_capabilities", return_value=frozenset({"jcl_evidence"})),
            patch(
                "cobol_rag.query.router_for",
                return_value=Mock(rank=lambda q, allowed=None: (CapabilityMatch("jcl_evidence", 0.69, 0.21),)),
            ),
        ):
            self.assertEqual(
                _absent_capability_request(
                    "Is there any batch JCL associated with PDCBVC?",
                    Mock(),
                    QueryScope(intent="general", program="PDCBVC"),
                ),
                "jcl_evidence",
            )

    def test_an_unconfident_match_does_not_claim_absence(self) -> None:
        with (
            patch("cobol_rag.query.unavailable_capabilities", return_value=frozenset({"jcl_evidence"})),
            patch(
                "cobol_rag.query.router_for",
                return_value=Mock(rank=lambda q, allowed=None: (CapabilityMatch("jcl_evidence", 0.31, 0.001),)),
            ),
        ):
            self.assertIsNone(_absent_capability_request(
                "What does this program do?",
                Mock(),
                QueryScope(intent="general", program="PDCBVC"),
            ))

    def test_a_confident_match_on_present_evidence_is_left_to_normal_routing(self) -> None:
        with (
            patch("cobol_rag.query.unavailable_capabilities", return_value=frozenset({"jcl_evidence"})),
            patch(
                "cobol_rag.query.router_for",
                return_value=Mock(rank=lambda q, allowed=None: (CapabilityMatch("cics_evidence", 0.82, 0.10),)),
            ),
        ):
            self.assertIsNone(_absent_capability_request(
                "Which CICS commands does PDCBVC issue?",
                Mock(),
                QueryScope(intent="general", program="PDCBVC"),
            ))

    def test_absence_answer_carries_the_analysis_reason_and_its_source(self) -> None:
        with patch(
            "cobol_rag.final_scripts_answers.capability_manifest", return_value=self.MANIFEST,
        ):
            answer = absent_capability_answer("PDCBVC", "jcl_evidence")

        self.assertIsNotNone(answer)
        self.assertIn("PDCBVC has no jcl evidence in the analyzed artifacts", answer)
        self.assertNotIn("evidence evidence", answer)
        self.assertIn("no program-to-JCL linkage in the parsed job steps", answer)
        self.assertIn("program.capability_manifest.json", answer)

    def test_available_capability_produces_no_absence_answer(self) -> None:
        with patch(
            "cobol_rag.final_scripts_answers.capability_manifest", return_value=self.MANIFEST,
        ):
            self.assertIsNone(absent_capability_answer("PDCBVC", "program_summary"))


class CompoundAspectSelectionTest(unittest.TestCase):
    """One question can ask about two aspects of the same variable."""

    @staticmethod
    def _matches(*pairs: tuple[str, float]) -> tuple[CapabilityMatch, ...]:
        ordered = sorted(pairs, key=lambda item: -item[1])
        return tuple(
            CapabilityMatch(
                name,
                score,
                (score - ordered[index + 1][1]) if index == 0 and index + 1 < len(ordered) else 0.0,
            )
            for index, (name, score) in enumerate(ordered)
        )

    def test_two_aspects_that_stand_alone_are_both_kept(self) -> None:
        # "what values can it take, and how does each affect execution" asks
        # about assigned constants and resulting control flow at once. Requiring
        # the leader to beat the runner-up rejected both.
        matches = self._matches(
            ("control_outcome", 0.6266),
            ("literal_assignments", 0.5916),
            ("variable_comparison", 0.5079),
        )

        self.assertEqual(
            confident_aspects(matches), ("control_outcome", "literal_assignments"),
        )

    def test_a_close_runner_up_below_the_bar_is_still_treated_as_noise(self) -> None:
        # A follow-up about reads belongs to no aspect. Its runner-up is close
        # but never confident on its own, so the pair must not be admitted.
        matches = self._matches(
            ("variable_definition", 0.5595),
            ("variable_comparison", 0.5180),
            ("literal_assignments", 0.4643),
        )

        self.assertEqual(confident_aspects(matches), ())

    def test_a_clear_single_aspect_is_unchanged(self) -> None:
        matches = self._matches(
            ("literal_assignments", 0.6319),
            ("variable_definition", 0.5530),
            ("variable_comparison", 0.5297),
        )

        self.assertEqual(confident_aspects(matches), ("literal_assignments",))

    def test_nothing_above_the_floor_selects_nothing(self) -> None:
        matches = self._matches(
            ("variable_lineage", 0.41), ("literal_assignments", 0.40),
        )

        self.assertEqual(confident_aspects(matches), ())

    def test_inbound_composition_is_an_aspect_a_question_can_ask_for(self) -> None:
        # Without it, "what is this field built from" could only be answered by
        # write sites, which name statements but never the contributing fields.
        self.assertIn("variable_composition", VARIABLE_ASPECT_DESCRIPTORS)
        self.assertIn(
            "variable_composition", _CAPABILITY_TASKS["variable_lineage"],
        )


class PerItemRequestsKeepTheirFormatterTest(unittest.TestCase):
    """Asking for detail on each item must not disqualify a list formatter."""

    @staticmethod
    def _plan(tasks: tuple[str, ...], intent: str) -> QueryPlan:
        return QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent=intent, domain="integration", tasks=tasks,
            relations=("example_per_item",),
            output_fields=("target", "paragraph", "commarea", "source_line", "length"),
        )

    def test_call_inventory_still_uses_its_formatter(self) -> None:
        # The formatter already lists every call with its own fields, which is
        # what the request asks for. Refusing it sent a well-specified inventory
        # question to free-form generation, which answered with uncited JSON.
        self.assertTrue(_direct_handler_supports(
            self._plan(("external_calls",), "external_programs"),
        ))

    def test_other_per_item_inventories_are_covered_too(self) -> None:
        for tasks, intent in (
            (("cics_operations",), "cics_operations"),
            (("copybook_inventory",), "copybooks"),
            (("literal_assignments",), "static_values"),
        ):
            with self.subTest(tasks=tasks):
                self.assertTrue(_direct_handler_supports(self._plan(tasks, intent)))

    def test_a_task_with_no_per_item_formatter_is_still_refused(self) -> None:
        self.assertFalse(_direct_handler_supports(
            self._plan(("path_from_paragraph",), "control_flow"),
        ))
        self.assertFalse(_direct_handler_supports(
            self._plan(("program_summary",), "program_summary"),
        ))


class SectionContractAcceptsRendererVocabularyTest(unittest.TestCase):
    """A contract checks for the finding, not for one renderer's heading."""

    RULE_ANSWER = "\n".join([
        "PDCBVC business rules (1 matching unique direct-evidence rule(s)):",
        "- BR-018",
        "  Condition: PXCSEMAF-OUTCOME NOT = SPACE",
        "  Action: CALL -> ABEND00",
        "  Paragraph: READ-TAB-SEMAF",
        "  Source location: PDCBVC.CBL line 765",
    ])

    @staticmethod
    def _plan(value: str, **kwargs) -> QueryPlan:
        return QueryPlan(
            program="PDCBVC", programs=("PDCBVC",),
            entities=(EntityReference("PDCBVC", "variable", value, f"PDCBVC|VARIABLE|{value}"),),
            **kwargs,
        )

    def test_control_outcome_is_satisfied_by_a_rule_action(self) -> None:
        # The business-rule renderer states the outcome as "Action: CALL -> X".
        # Demanding one heading discarded this cited, line-accurate answer.
        plan = self._plan(
            "PXCSEMAF-OUTCOME", intent="business_rules", domain="control_flow",
            tasks=("control_outcome",),
        )

        self.assertTrue(validate_plan_answer(plan, self.RULE_ANSWER).passed)

    def test_an_answer_with_no_outcome_at_all_is_still_rejected(self) -> None:
        plan = self._plan(
            "PXCSEMAF-OUTCOME", intent="business_rules", domain="control_flow",
            tasks=("control_outcome",),
        )

        contract = validate_plan_answer(plan, "PXCSEMAF-OUTCOME is checked somewhere.")

        self.assertFalse(contract.passed)
        self.assertIn("missing_requested_section:control_outcome", contract.reasons)

    def test_exhaustiveness_is_satisfied_by_a_coverage_ratio(self) -> None:
        # "5/5 direct site(s) returned" is the same completeness claim as an
        # item count, and is the only form the variable renderer emits.
        plan = self._plan(
            "M1MSGO", intent="variable_dataflow", domain="dataflow",
            tasks=("variable_writes", "literal_assignments"), result_scope="all",
        )
        answer = "\n".join([
            "M1MSGO direct COBOL access evidence:",
            "Modified at:",
            "- BROWSE-FASE1, line 285: MOVE KFINE-DATI TO M1MSGO.",
            "Write coverage: 5/5 direct site(s) returned.",
        ])

        self.assertTrue(validate_plan_answer(plan, answer).passed)

    def test_an_exhaustive_answer_claiming_no_completeness_is_still_rejected(self) -> None:
        plan = self._plan(
            "M1MSGO", intent="variable_dataflow", domain="dataflow",
            tasks=("variable_writes", "literal_assignments"), result_scope="all",
        )

        contract = validate_plan_answer(plan, "M1MSGO is assigned in several paragraphs.")

        self.assertFalse(contract.passed)
        self.assertIn("missing_exhaustive_result_count", contract.reasons)

    def test_item_count_is_still_cross_checked_against_the_listed_lines(self) -> None:
        plan = QueryPlan(
            program="PDCBVC", intent="static_values", tasks=("literal_assignments",),
            result_scope="all",
        )
        answer = "Literal assignments: 3 matching item(s).\n- line 226: X\n- line 288: Y"

        self.assertIn(
            "exhaustive_result_count_mismatch:2/3",
            validate_plan_answer(plan, answer).reasons,
        )

    def test_truncation_is_still_refused_however_completeness_is_worded(self) -> None:
        plan = self._plan(
            "M1MSGO", intent="variable_dataflow", domain="dataflow",
            tasks=("variable_writes",), result_scope="all",
        )
        answer = "M1MSGO modified at:\nshowing only the first 5 items.\nWrite coverage: 5/5 direct site(s) returned."

        self.assertIn("exhaustive_result_truncated", validate_plan_answer(plan, answer).reasons)


class OffEntityClaimReroutingTest(unittest.TestCase):
    """A claim whose evidence is off-entity is retried against another capability."""

    @staticmethod
    def _plan(entity_type: str = "variable") -> QueryPlan:
        entity = EntityReference(
            "PDCBVC", entity_type, "NPAGT", "PDCBVC|VARIABLE|NPAGT",
        )
        return QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="business_rules",
            domain="control_flow", tasks=("business_rules",), entities=(entity,),
        )

    @staticmethod
    def _subtask(capability: str) -> EvidenceSubtask:
        return EvidenceSubtask(
            claim_id="claim_1",
            description=f"Verify {capability} evidence for NPAGT",
            capability=capability,
            tasks=("business_rules",),
            entity_values=("NPAGT",),
            source_domains=("business_rule",),
            relations=("before",),
        )

    def test_reroute_only_considers_capabilities_that_index_the_entity_type(self) -> None:
        captured: dict[str, object] = {}

        def rank(question: str, *, allowed=None):
            captured["allowed"] = set(allowed)
            return (CapabilityMatch("variable_access", 0.81, 0.09),)

        with patch("cobol_rag.query.router_for", return_value=Mock(rank=rank)):
            rerouted = _rerouted_subtask(
                question="Which paragraphs modify NPAGT?",
                config=Mock(),
                plan=self._plan(),
                subtask=self._subtask("condition_outcome"),
                tried=frozenset({"condition_outcome"}),
            )

        self.assertIsNotNone(rerouted)
        self.assertEqual(rerouted.capability, "variable_access")
        self.assertEqual(rerouted.tasks, _CAPABILITY_TASKS["variable_access"])
        # The claim keeps the identifier the user typed, and drops the source
        # domains and relations that belonged to the capability that failed.
        self.assertEqual(rerouted.entity_values, ("NPAGT",))
        self.assertEqual(rerouted.source_domains, ())
        self.assertEqual(rerouted.relations, ())
        # Already-tried and entity-incompatible capabilities are never offered.
        self.assertNotIn("condition_outcome", captured["allowed"])
        self.assertNotIn("program_summary", captured["allowed"])
        self.assertNotIn("cics_evidence", captured["allowed"])

    def test_claims_without_entities_are_not_rerouted(self) -> None:
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="program_summary",
            domain="program_structure", tasks=("program_summary",),
        )

        self.assertIsNone(_rerouted_subtask(
            question="What is PDCBVC?",
            config=Mock(),
            plan=plan,
            subtask=self._subtask("program_summary"),
            tried=frozenset({"program_summary"}),
        ))

    def test_reroute_stops_when_every_capability_has_been_tried(self) -> None:
        exhausted = frozenset(_CAPABILITY_ENTITY_TYPES)

        self.assertIsNone(_rerouted_subtask(
            question="Which paragraphs modify NPAGT?",
            config=Mock(),
            plan=self._plan(),
            subtask=self._subtask("condition_outcome"),
            tried=exhausted,
        ))

    def test_off_entity_artifact_ends_the_attempt_before_any_model_call(self) -> None:
        # Re-routing must stay cheap: once the artifact is shown to be about
        # something else, retrieval and generation would read the same records,
        # so neither may run while another route is still available.
        get_llm = Mock()
        with (
            patch("cobol_rag.query._direct_handler_supports", return_value=True),
            patch(
                "cobol_rag.query.answer_from_final_scripts",
                return_value="PDCBVC business rules (32 matching rule(s)):\n- BR-001 TWCOB-FASE\n",
            ),
            patch("cobol_rag.query._final_script_sources", return_value=[]),
            patch("cobol_rag.query._retrieve_for_plan") as retrieve_for_plan,
        ):
            result = _attempt_evidence_subtask(
                question="Which paragraphs modify NPAGT?",
                config=Mock(),
                parent_plan=self._plan(),
                parent_scope=QueryScope(intent="business_rules"),
                subtask=self._subtask("condition_outcome"),
                top_k=None,
                chunk_types=None,
                conversation_history=None,
                get_llm=get_llm,
                reroute_available=True,
            )

        self.assertFalse(result.passed)
        self.assertIn("missing_requested_entity:NPAGT", result.reasons)
        retrieve_for_plan.assert_not_called()
        get_llm.assert_not_called()

    def test_final_attempt_still_runs_the_full_fallback_chain(self) -> None:
        # With no route left, behaviour matches not re-routing at all, so the
        # bound on re-routing never costs an answer that the chain would find.
        with (
            patch("cobol_rag.query._direct_handler_supports", return_value=True),
            patch(
                "cobol_rag.query.answer_from_final_scripts",
                return_value="PDCBVC business rules (32 matching rule(s)):\n- BR-001 TWCOB-FASE\n",
            ),
            patch("cobol_rag.query._final_script_sources", return_value=[]),
            patch("cobol_rag.query._retrieve_for_plan") as retrieve_for_plan,
        ):
            retrieve_for_plan.return_value = Mock(
                results=[], guard=Mock(status="insufficient", reasons=("no_evidence",)),
            )
            _attempt_evidence_subtask(
                question="Which paragraphs modify NPAGT?",
                config=Mock(),
                parent_plan=self._plan(),
                parent_scope=QueryScope(intent="business_rules"),
                subtask=self._subtask("condition_outcome"),
                top_k=None,
                chunk_types=None,
                conversation_history=None,
                get_llm=Mock(),
                reroute_available=False,
            )

        retrieve_for_plan.assert_called_once()

    def test_only_an_off_entity_failure_triggers_a_reroute(self) -> None:
        plan = self._plan()
        subtask = self._subtask("condition_outcome")

        off_entity = EvidenceSubtaskResult(
            subtask=subtask, plan=plan, passed=False,
            reasons=("missing_requested_entity:NPAGT",),
        )
        other_failure = EvidenceSubtaskResult(
            subtask=subtask, plan=plan, passed=False,
            reasons=("no_sufficient_subtask_evidence",),
        )
        success = EvidenceSubtaskResult(
            subtask=subtask, plan=plan, passed=True, answer="grounded",
        )

        self.assertTrue(_entity_contract_failed(off_entity))
        self.assertFalse(_entity_contract_failed(other_failure))
        self.assertFalse(_entity_contract_failed(success))


if __name__ == "__main__":
    unittest.main()
