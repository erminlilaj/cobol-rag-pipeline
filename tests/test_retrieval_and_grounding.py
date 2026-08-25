from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import ANY, Mock, patch

from cobol_rag.chat import ChatSession, ChatTurn
from cobol_rag.config import AppConfig
from cobol_rag.loaders.rag_documents import RagDocumentsLoader
from cobol_rag.capability_router import (
    CAPABILITY_DESCRIPTORS,
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
    _claim_should_be_rerouted,
    _entity_contract_failed,
    _rerouted_subtask,
    _refine_variable_tasks,
    _supplement_missing_capability,
    _assess_technical_answerability,
    EvidenceSubtaskResult,
    _decision_respects_candidates,
    _routing_candidates,
    _chunk_types_for_plan,
    _complete_with_transient_retry,
    _conversational_route_is_blocked,
    _compose_subtask_results,
    _deterministic_routing,
    _direct_handler_supports,
    _ensure_citations,
    _effective_subtask_contract_plan,
    _parse_routing_decision,
    _retrieve_for_plan,
    _render_verified_candidate_to_contract,
    _repair_conversational_reply,
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
    _is_explicit_followup,
    _CAPABILITY_ENTITY_TYPES,
    _CAPABILITY_TASKS,
    EvidenceSubtask,
    QueryPlan,
    ResponseContract,
    build_query_plan,
    detect_message_language,
    derive_evidence_subtasks,
    execution_strategy_for_plan,
    merge_semantic_plan,
    parse_response_contract,
    plan_for_subtask,
    resolve_response_language,
    validate_plan_answer,
    validate_evidence_answer,
)
from cobol_rag.final_scripts_answers import (
    _answer_artifact_inventory,
    _answer_cics_operations,
    _cics_statement_resources,
    absent_capability_answer,
    answer_source_line_spans,
    answer_source_lines,
)
from cobol_rag.index import compose_prose
from cobol_rag.retrieve import (
    EvidenceGuard, RetrievalOutcome, RetrievalResult, _deduplicate_results, _detect_intent,
    _expand_context, _lexical_from_collection, clear_retrieval_cache,
    detect_intent_with_basis,
)
from cobol_rag.scope import (
    EntityReference,
    QueryScope,
    SessionState,
    resolve_query_scope,
    source_address_entity,
    refers_to_previous_turn,
    source_address_in,
    source_addresses_in,
)


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
    def test_response_contract_parses_shape_limits_and_exact_counts(self) -> None:
        sentence = parse_response_contract("Describe PDCBVC in one sentence only and at most 20 words.")
        self.assertEqual(sentence.format, "sentence")
        self.assertEqual(sentence.max_sentences, 1)
        self.assertEqual(sentence.max_words, 20)
        self.assertTrue(sentence.only_requested_content)

        bullets = parse_response_contract("Give exactly three bullet points.")
        self.assertEqual(bullets.format, "bullets")
        self.assertEqual(bullets.exact_item_count, 3)

        first_three = parse_response_contract("Return exactly the first 3 literal assignments.")
        self.assertEqual(first_three.exact_item_count, 3)

        json_array = parse_response_contract("Return a JSON array only.")
        self.assertEqual(json_array.format, "json_array")

        two_lines = parse_response_contract("Explain PDCBVC in two lines.")
        self.assertEqual(two_lines.max_lines, 2)
        self.assertEqual(two_lines.exact_lines, 2)

        bounded_lines = parse_response_contract("Explain PDCBVC in at most three lines.")
        self.assertEqual(bounded_lines.max_lines, 3)
        self.assertIsNone(bounded_lines.exact_lines)

        named_variables = parse_response_contract("Name 10 variables inside PDCBVC.")
        self.assertEqual(named_variables.exact_item_count, 10)

        variable_count = parse_response_contract("How many variables are in PDCBVC?")
        self.assertEqual(variable_count.format, "count")

        compound_count = parse_response_contract(
            "Give me a summary of PDCBVC and tell me how many variables it has."
        )
        self.assertEqual(compound_count.format, "default")

    def test_lexical_collection_reads_only_the_selected_program_and_caches_it(self) -> None:
        collection = Mock()
        collection.get.return_value = {
            "documents": ["PDCBVC variable WCTRIG"],
            "metadatas": [{"program": "PDCBVC", "chunk_type": "dataflow.variable"}],
        }
        resources = Mock(chroma_collection=collection)
        clear_retrieval_cache()
        with patch("cobol_rag.retrieve.open_index", return_value=resources):
            first = _lexical_from_collection(
                "WCTRIG", AppConfig(), 5, None, {"program": "PDCBVC"},
            )
            second = _lexical_from_collection(
                "WCTRIG", AppConfig(), 5, None, {"program": "PDCBVC"},
            )
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        collection.get.assert_called_once_with(
            include=["documents", "metadatas"], where={"program": "PDCBVC"},
        )

    def test_parent_expansion_walks_entity_to_domain_and_program_nodes(self) -> None:
        collection = Mock()
        collection.get.return_value = {
            "documents": ["Variable domain summary", "Program summary"],
            "metadatas": [
                {
                    "program": "PDCBVC", "node_id": "domain:PDCBVC:variable_dataflow",
                    "hierarchy_level": "domain", "intent_domain": "variable_dataflow",
                    "chunk_type": "dataflow.used_variables", "source_id": "domain",
                },
                {
                    "program": "PDCBVC", "node_id": "program:PDCBVC",
                    "hierarchy_level": "program", "intent_domain": "program_summary",
                    "chunk_type": "program.summary", "source_id": "program",
                },
            ],
        }
        primary = RetrievalResult(
            1.0,
            "WCTRIG evidence",
            {
                "program": "PDCBVC", "source_id": "entity",
                "node_id": "entity:PDCBVC|VARIABLE|WCTRIG",
                "parent_id": "domain:PDCBVC:variable_dataflow",
                "domain_parent_id": "domain:PDCBVC:variable_dataflow",
                "program_parent_id": "program:PDCBVC",
            },
        )
        clear_retrieval_cache()
        resources = Mock(chroma_collection=collection)
        with patch("cobol_rag.retrieve.open_index", return_value=resources):
            expanded = _expand_context(
                [primary], AppConfig(), program="PDCBVC", entity_key=None,
                entity_value="WCTRIG", intent="variable_dataflow",
            )
        self.assertEqual({item.metadata["source_id"] for item in expanded}, {"domain", "program"})
        self.assertTrue(all(item.metadata["context_role"] == "parent_context" for item in expanded))

    def test_response_contract_validation_is_mechanical(self) -> None:
        one_sentence = QueryPlan(response_contract=ResponseContract(format="sentence", max_sentences=1))
        self.assertTrue(validate_plan_answer(one_sentence, "One supported sentence.").passed)
        invalid = validate_plan_answer(one_sentence, "First sentence. Second sentence.")
        self.assertIn("max_sentences_exceeded:2/1", invalid.reasons)

        three_bullets = QueryPlan(
            response_contract=ResponseContract(format="bullets", exact_item_count=3),
        )
        self.assertTrue(validate_plan_answer(three_bullets, "- one\n- two\n- three").passed)
        self.assertIn(
            "exact_item_count_mismatch:2/3",
            validate_plan_answer(three_bullets, "- one\n- two").reasons,
        )

    def test_high_confidence_plan_rejects_llm_capability_expansion(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = build_query_plan(
            "Explain PDCBVC in three lines.", scope, intent="general",
        )
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "program_summary",
            "tasks": ["program_summary", "complete_program_flow"],
            "source_domains": ["program.summary", "controlflow.cfg"],
            "subtasks": [
                {"capability": "program_summary", "tasks": ["program_summary"]},
                {"capability": "control_flow", "tasks": ["complete_program_flow"]},
            ],
            "confidence": 0.99,
        })

        self.assertEqual(merged.tasks, ("program_summary",))
        self.assertEqual([item.capability for item in merged.subtasks], ["program_summary"])
        self.assertIn("task_not_authorized:complete_program_flow", merged.policy_rejections)
        self.assertEqual(execution_strategy_for_plan(merged), "single_claim")

    def test_subtask_keeps_response_contract_but_evidence_validation_ignores_shape(self) -> None:
        plan = QueryPlan(
            program="PDCBVC",
            programs=("PDCBVC",),
            intent="program_summary",
            tasks=("program_summary",),
            response_contract=ResponseContract(max_lines=2),
        )
        subtask = EvidenceSubtask(
            claim_id="claim_1",
            description="Summarize PDCBVC",
            capability="program_summary",
            tasks=("program_summary",),
        )
        subplan = plan_for_subtask(plan, subtask)
        four_lines = "One.\nTwo.\nThree.\nFour."

        self.assertEqual(subplan.response_contract.max_lines, 2)
        self.assertFalse(validate_plan_answer(subplan, four_lines).passed)
        self.assertTrue(validate_evidence_answer(subplan, four_lines).passed)

    def test_verified_candidate_is_rendered_to_final_response_contract(self) -> None:
        source = RetrievalResult(
            1.0,
            "PDCBVC is a COBOL CICS program. It has analyzed calls and control flow.",
            {
                "source_id": "summary-1",
                "source_file": "program.summary.json",
                "chunk_type": "program.summary",
                "program": "PDCBVC",
            },
        )
        llm = Mock()
        llm.complete.return_value = Mock(text=(
            "PDCBVC is a COBOL CICS program. [Source 1]\n"
            "Its analyzed evidence includes calls and control flow. [Source 1]"
        ))
        resources = Mock()
        resources.runtime.llm = llm
        plan = QueryPlan(
            program="PDCBVC",
            programs=("PDCBVC",),
            intent="program_summary",
            tasks=("program_summary",),
            response_contract=ResponseContract(max_lines=2),
        )
        with patch("cobol_rag.query.open_index", return_value=resources):
            rendered = _render_verified_candidate_to_contract(
                question="Explain PDCBVC in two lines.",
                candidate="One.\nTwo.\nThree.",
                sources=[source],
                plan=plan,
                config=AppConfig(),
            )

        self.assertTrue(rendered.passed)
        self.assertEqual(len(rendered.answer.splitlines()), 2)
        self.assertEqual(llm.complete.call_count, 1)

    def test_presentation_return_is_not_a_cobol_return_operation(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="external_programs")
        plan = build_query_plan(
            "List PDCBVC external calls; return only target and call type.",
            scope,
            intent="external_programs",
        )
        self.assertNotIn("RETURN", plan.operations)
        self.assertEqual(set(plan.output_fields), {"target", "call_type"})
        self.assertTrue(plan.only_requested_fields)

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

    def test_final_contract_uses_successful_rerouted_capability_tasks(self) -> None:
        entity = EntityReference(
            "PDCBVC", "variable", "STATUS-FIELD", "PDCBVC|VARIABLE|STATUS-FIELD",
        )
        original = EvidenceSubtask(
            claim_id="claim_1",
            description="Show the condition driven by STATUS-FIELD",
            capability="condition_outcome",
            tasks=("control_outcome",),
            entity_values=("STATUS-FIELD",),
        )
        rerouted = replace(
            original,
            capability="variable_access",
            tasks=("variable_reads", "variable_writes"),
        )
        parent = QueryPlan(
            program="PDCBVC",
            programs=("PDCBVC",),
            intent="variable_dataflow",
            tasks=("control_outcome",),
            entities=(entity,),
            subtasks=(original,),
        )
        result = EvidenceSubtaskResult(
            subtask=rerouted,
            plan=plan_for_subtask(parent, rerouted),
            passed=True,
            answer=(
                "STATUS-FIELD direct evidence:\n"
                "Modified at: line 10.\n"
                "Tested/read at: line 20."
            ),
        )

        effective = _effective_subtask_contract_plan(parent, (result,))

        self.assertNotIn("control_outcome", effective.tasks)
        self.assertEqual(effective.tasks, ("variable_reads", "variable_writes"))
        self.assertFalse(validate_plan_answer(parent, result.answer).passed)
        self.assertTrue(validate_plan_answer(effective, result.answer).passed)

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

    def test_a_language_switch_is_recognised_without_enumerating_verbs(self) -> None:
        # Observed production failure: "no facciamo in englese" left the session
        # locked in Italian because the request patterns enumerated verbs
        # (rispondi/scrivi/parla/continua) and knew neither "facciamo" nor
        # "parliamo". Naming a language is the request, whatever verb carries it,
        # and a near match survives the misspelling people actually type.
        italian = SessionState(response_language="it")
        for message in (
            "no facciamo in englese",
            "no facciamo in inglese",
            "parliamo in inglese",
            "switch to english",
            "meglio in inglese",
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    resolve_response_language(message, italian), ("en", "explicit_request"),
                )
        self.assertEqual(
            resolve_response_language("torniamo all'italiano", SessionState(response_language="en")),
            ("it", "explicit_request"),
        )

    def test_a_language_name_inside_an_identifier_is_not_a_language_request(self) -> None:
        # The decoys that keep the rule above from swallowing COBOL questions:
        # a language name welded into an identifier is evidence, not a preference.
        italian = SessionState(response_language="it")
        for message in (
            "What does ENGLISH-FLAG do?",
            "Where is WS-ITALIAN-NAME written?",
            "Show me lines 10-20 of PDCBVC.",
        ):
            with self.subTest(message=message):
                language, source = resolve_response_language(message, italian)
                # These resolve by the language the question is written in. What
                # must never happen is the identifier being read as a preference:
                # WS-ITALIAN-NAME does not order Italian replies.
                self.assertNotEqual(source, "explicit_request")
                self.assertEqual(language, "en")

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
        plan = build_query_plan("hi", QueryScope(), state=SessionState(response_language="it"))
        with patch("cobol_rag.query.build_llm") as build, \
                patch("cobol_rag.query.compose_prose") as compose:
            build.return_value.complete.return_value = wrong
            compose.return_value = "<reply>Hello! How are you?</reply>"
            decision = _route_query(
                "hi", AppConfig(), session_state=SessionState(response_language="it"),
                preliminary_plan=plan, preliminary_scope=QueryScope(),
            )
        self.assertEqual(compose.call_count, 1)
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

    def test_start_to_finish_wording_is_complete_flow_not_a_named_start_path(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="control_flow")
        plan = build_query_plan(
            "Walk me through what PDCBVC does from start to finish.",
            scope,
            intent="control_flow",
        )
        self.assertEqual(plan.tasks, ("complete_program_flow",))
        self.assertNotIn("starts_at", plan.relations)

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

    def test_call_result_phrase_triggers_post_call_context(self) -> None:
        entity = EntityReference("PDCBVC", "call", "PXRSEMAF", "PDCBVC|PXRSEMAF|CALL")
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            intent="external_programs",
        )
        plan = build_query_plan(
            "What result of the PXRSEMAF call is checked by PDCBVC?",
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

    def test_variable_catalogue_preserves_count_limit_and_control_filter(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="variable_inventory")
        limited = build_query_plan(
            "Name 10 variables inside PDCBVC.", scope, intent="variable_inventory",
        )
        self.assertEqual(limited.response_contract.exact_item_count, 10)

        counted = build_query_plan(
            "How many variables are inside PDCBVC?", scope, intent="variable_inventory",
        )
        self.assertEqual(counted.response_contract.format, "count")

        filtered = build_query_plan(
            "Which variables control the flow in PDCBVC?", scope, intent="variable_inventory",
        )
        self.assertIn("control_usage", filtered.output_fields)
        self.assertEqual(filtered.result_scope, "all")

        sampled = build_query_plan(
            "Give me a sample of the fields used in PDCBVC.",
            scope,
            intent="variable_inventory",
        )
        self.assertEqual(sampled.response_contract.exact_item_count, 10)

    def test_compound_program_summary_and_variable_count_preserves_both_claims(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="variable_inventory")
        plan = build_query_plan(
            "Give me a summary of PDCBVC and tell me how many variables it has.",
            scope,
            intent="variable_inventory",
        )
        self.assertEqual(plan.intent, "program_summary")
        self.assertEqual(plan.category, "multi_source_synthesis")
        self.assertEqual(set(plan.tasks), {"program_summary", "variable_inventory"})
        self.assertIn("item_count", plan.output_fields)
        self.assertEqual(
            {subtask.capability for subtask in plan.subtasks},
            {"program_summary", "variable_inventory"},
        )
        self.assertEqual(execution_strategy_for_plan(plan), "agentic")

        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "program_summary",
            "tasks": ["program_summary", "variable_inventory"],
            "subtasks": [
                {
                    "capability": "program_summary",
                    "tasks": ["program_summary"],
                    "output_fields": [],
                },
                {
                    "capability": "variable_inventory",
                    "tasks": ["variable_inventory"],
                    "output_fields": ["variables"],
                },
            ],
        })
        variable_claim = next(
            subtask for subtask in merged.subtasks
            if subtask.capability == "variable_inventory"
        )
        self.assertIn("item_count", variable_claim.output_fields)

    def test_subtask_composer_hides_internal_claim_language(self) -> None:
        summary_task = EvidenceSubtask(
            claim_id="claim_1",
            description="Summarize the program for PDCBVC",
            capability="program_summary",
            tasks=("program_summary",),
        )
        variables_task = EvidenceSubtask(
            claim_id="claim_2",
            description="Inspect the variable catalogue for PDCBVC",
            capability="variable_inventory",
            tasks=("variable_inventory",),
        )
        results = (
            EvidenceSubtaskResult(
                subtask=summary_task,
                plan=QueryPlan(tasks=("program_summary",)),
                passed=True,
                answer="PDCBVC is an analyzed COBOL CICS program.",
            ),
            EvidenceSubtaskResult(
                subtask=variables_task,
                plan=QueryPlan(tasks=("variable_inventory",)),
                passed=True,
                answer="168",
            ),
        )
        answer, _ = _compose_subtask_results(results)
        self.assertIn("Program summary", answer)
        self.assertIn("Variable count\n168 analyzed variables.", answer)
        self.assertNotIn("Claim 1", answer)
        self.assertNotIn("Verify", answer)

    def test_variable_continuation_advances_past_the_previous_page(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        state = SessionState(
            current_program="PDCBVC",
            current_intent="variable_inventory",
            current_plan={
                "intent": "variable_inventory",
                "result_offset": 0,
                "response_contract": {"exact_item_count": 25},
            },
        )
        plan = build_query_plan(
            "There is more, show me the rest.", scope, intent="general", state=state,
        )
        self.assertEqual(plan.intent, "variable_inventory")
        self.assertEqual(plan.result_offset, 25)

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

    def test_access_and_lineage_for_one_variable_collapse_into_one_execution(self) -> None:
        entity = EntityReference("PDCBVC", "variable", "NPAGT", "PDCBVC|VARIABLE|NPAGT")
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), entities=(entity,),
            intent="variable_dataflow",
        )
        plan = build_query_plan(
            "Tell me everything about NPAGT in PDCBVC.", scope,
            intent="variable_dataflow",
        )
        self.assertIn("variable_lineage", plan.tasks)
        matching = [
            item for item in plan.subtasks
            if item.capability in {"variable_access", "variable_lineage"}
        ]
        self.assertEqual(len(matching), 1)

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

    def test_describe_named_program_builds_a_typed_summary_plan(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = build_query_plan(
            "Describe PDCBVC in one sentence only.", scope, intent="general",
        )
        self.assertEqual(plan.intent, "program_summary")
        self.assertEqual(plan.tasks, ("program_summary",))
        self.assertEqual(plan.response_contract.max_sentences, 1)

    def test_what_is_named_program_builds_summary_instead_of_dataset_plan(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="datasets_tables")
        plan = build_query_plan("What is PDCBVC?", scope, intent="datasets_tables")
        self.assertEqual(plan.intent, "program_summary")
        self.assertEqual(plan.tasks, ("program_summary",))
        self.assertEqual(plan.operations, ("describe",))

    def test_named_program_file_existence_is_not_jcl_file_io(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="datasets_tables")
        plan = build_query_plan(
            "You have a file called PDCBVC?", scope, intent="datasets_tables",
        )
        self.assertEqual(plan.intent, "program_summary")
        self.assertEqual(plan.tasks, ("program_summary",))
        self.assertIn("exists", plan.operations)
        self.assertNotIn("jcl.file_io", plan.source_domains)

    def test_two_lines_is_response_shape_not_source_location(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = build_query_plan("Explain PDCBVC in two lines.", scope, intent="general")
        self.assertEqual(plan.intent, "program_summary")
        self.assertEqual(plan.response_contract.max_lines, 2)
        self.assertNotIn("source_line", plan.output_fields)

    def test_conflicting_semantic_tasks_cannot_poison_typed_summary_plan(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = build_query_plan(
            "Describe PDCBVC in one sentence only.", scope, intent="general",
        )
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "variable_dataflow",
            "domain": "dataflow",
            "tasks": ["variable_definition", "variable_reads", "variable_writes"],
            "subtasks": [{
                "capability": "variable_access",
                "tasks": ["variable_definition", "variable_reads", "variable_writes"],
            }],
            "confidence": 0.6,
        })
        self.assertEqual(merged.intent, "program_summary")
        self.assertEqual(merged.tasks, ("program_summary",))
        self.assertEqual(tuple(task.capability for task in merged.subtasks), ("program_summary",))

    def test_literal_assignment_wording_builds_static_value_plan(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = build_query_plan(
            "Return exactly the first 3 literal assignments in PDCBVC.",
            scope,
            intent="general",
        )
        self.assertEqual(plan.intent, "static_values")
        self.assertEqual(plan.tasks, ("literal_assignments",))
        self.assertEqual(plan.response_contract.exact_item_count, 3)

    def test_leading_return_is_presentation_not_cics_operation(self) -> None:
        scope = QueryScope(
            program="PDCBVC", programs=("PDCBVC",), intent="external_programs",
        )
        plan = build_query_plan(
            "Return every external program called by PDCBVC as a JSON array with only target and call type.",
            scope,
            intent="external_programs",
        )
        self.assertNotIn("RETURN", plan.operations)
        self.assertTrue(plan.only_requested_fields)
        self.assertEqual(plan.output_fields, ("target", "call_type"))
        self.assertEqual(plan.response_contract.format, "json_array")

    def test_semantic_subtask_must_match_typed_variable_tasks(self) -> None:
        scope = QueryScope(
            program="PDCBVC",
            programs=("PDCBVC",),
            entities=(EntityReference(
                program="PDCBVC", entity_type="variable", value="WDATE2-GG",
                entity_key="PDCBVC|VARIABLE|WDATE2-GG",
            ),),
            intent="variable_dataflow",
        )
        plan = build_query_plan(
            "What is WDATE2-GG? Include its declaration and parent group.",
            scope,
            intent="variable_dataflow",
        )
        merged = merge_semantic_plan(plan, {
            "route": "technical",
            "intent": "variable_dataflow",
            "tasks": ["variable_definition"],
            "subtasks": [
                {"capability": "variable_access", "tasks": ["variable_definition"]},
                {"capability": "db2_evidence", "tasks": ["db2_tables"]},
            ],
        })
        self.assertNotIn("db2_evidence", tuple(task.capability for task in merged.subtasks))

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

    def test_conversational_prose_is_composed_apart_from_the_classification(self) -> None:
        # Observed production failure: asked in Italian whether it speaks Italian,
        # the assistant replied "Bene, tu?". The same model answers the question
        # correctly on its own — the prose was degraded by being the last field of
        # a routing object while the model chose a COBOL capability. Composition
        # therefore runs as its own call, and its result is what the user sees.
        routing = type("Response", (), {"text": json.dumps({
            "route": "conversational", "category": "conversational",
            "domain": "conversation", "intent": "general", "tasks": [], "relations": [],
            "response_language": "it", "confidence": 0.9, "reply": "Bene, tu?",
        })})()
        with patch("cobol_rag.query.build_llm") as build, \
                patch("cobol_rag.query.compose_prose") as compose:
            build.return_value.complete.return_value = routing
            compose.return_value = "Si, parlo italiano. Chiedimi pure quello che vuoi."
            decision = _route_query(
                "tu parli bene l'Italiano??", AppConfig(),
                preliminary_plan=QueryPlan(intent="general", response_language="it"),
                preliminary_scope=QueryScope(intent="general"),
            )
        self.assertEqual(decision.route, "conversational")
        self.assertEqual(decision.reply, "Si, parlo italiano. Chiedimi pure quello che vuoi.")

    def test_prose_is_generated_through_the_chat_template_not_completion(self) -> None:
        # The root cause of "Bene, tu?": Ollama's completion endpoint sends the
        # prompt with no instruct template, so an instruct model continues the
        # text instead of answering it. Measured on granite, the same Italian
        # question returns "Bene, tu?" through completion at every temperature
        # and every prompt shape, and answers correctly through chat. Structured
        # extraction survives completion because a JSON schema re-anchors the
        # model; prose does not, so prose must use chat.
        with patch("cobol_rag.index.build_llm") as build:
            build.return_value.chat.return_value = Mock(
                message=Mock(content="  Certo.  "),
            )
            out = compose_prose(AppConfig(), system="be brief", user="ciao")
        self.assertEqual(out, "Certo.")
        build.return_value.complete.assert_not_called()
        messages = build.return_value.chat.call_args.args[0]
        self.assertEqual([m.role.value for m in messages], ["system", "user"])
        self.assertEqual([m.content for m in messages], ["be brief", "ciao"])

    def test_an_unclassifiable_message_is_not_composed_into_free_prose(self) -> None:
        # Composing an "unclear" message breaks the scope boundary: asked for the
        # capital of France the model answers it, under every boundary
        # instruction measured. Only small talk is composed ahead of time.
        unclear = type("Response", (), {"text": json.dumps({
            "route": "unclear", "category": "unclear", "domain": "conversation",
            "intent": "general", "tasks": [], "relations": [],
            "response_language": "en", "confidence": 0.2,
            "reply": "I can only help with COBOL analysis.",
        })})()
        with patch("cobol_rag.query.build_llm") as build, \
                patch("cobol_rag.query.compose_prose") as compose:
            build.return_value.complete.return_value = unclear
            _route_query(
                "what's the capital of France?", AppConfig(),
                preliminary_plan=QueryPlan(intent="general"),
                preliminary_scope=QueryScope(intent="general"),
            )
        compose.assert_not_called()

    def test_the_composer_prompt_only_carries_the_clause_it_needs(self) -> None:
        # Every extra sentence measurably cost answer quality on this model, and
        # an imperative clause came back echoed inside the reply. The switch
        # clause therefore appears only when the session language actually
        # changes.
        with patch("cobol_rag.query.compose_prose") as compose:
            compose.return_value = "Understood."
            _repair_conversational_reply(
                "no facciamo in englese", route="conversational",
                required_language="en", previous_language="it", config=AppConfig(),
            )
        switched = compose.call_args.kwargs["system"]
        self.assertIn("English", switched)
        self.assertIn("asked for replies in English", switched)
        # Regression: composing from a bare "helpful assistant" prompt answered
        # "What is the weather in Rome today?" with a temperature, where the
        # planner's own reply had declined for want of live data.
        self.assertIn("no internet access", switched)

        with patch("cobol_rag.query.compose_prose") as compose:
            compose.return_value = "Hello!"
            _repair_conversational_reply(
                "hi", route="conversational",
                required_language="en", previous_language="en", config=AppConfig(),
            )
        steady = compose.call_args.kwargs["system"]
        self.assertNotIn("asked for replies", steady)

    def test_a_composer_that_answers_with_json_cannot_reach_the_user(self) -> None:
        # A model primed by the routing schema can answer the composer with a
        # routing object. Prose is never a bare JSON blob, so the classifier's own
        # reply is kept rather than showing the user a raw object.
        blob = json.dumps({
            "route": "conversational", "category": "conversational",
            "domain": "conversation", "intent": "general", "tasks": [], "relations": [],
            "response_language": "en", "confidence": 0.9, "reply": "Hello! How can I help?",
        })
        echoed = type("Response", (), {"text": blob})()
        with patch("cobol_rag.query.build_llm") as build, \
                patch("cobol_rag.query.compose_prose") as compose:
            build.return_value.complete.return_value = echoed
            compose.return_value = blob
            decision = _route_query(
                "hi", AppConfig(),
                preliminary_plan=QueryPlan(intent="general"),
                preliminary_scope=QueryScope(intent="general"),
            )
        self.assertEqual(decision.reply, "Hello! How can I help?")
        self.assertNotIn("{", decision.reply)

    def test_pagination_literal_constraint_builds_a_technical_task(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = build_query_plan(
            "Explain the pagination logic in PDCBVC using direct evidence.", scope, intent="general",
        )
        self.assertEqual(plan.intent, "control_flow")
        self.assertEqual(plan.tasks, ("pagination_logic",))

    def test_natural_paging_phrase_builds_the_same_pagination_task(self) -> None:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        plan = build_query_plan(
            "Explain how PDCBVC moves through the result pages.", scope, intent="general",
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

    def test_a_flat_cluster_behind_a_weak_leader_selects_nothing(self) -> None:
        # Three aspects within a thousandth of each other, all just over the
        # floor, is a question that matched nothing clearly. Reading them as
        # three requests produced a claim whose evidence did not exist, so the
        # leader has to be confident before it can introduce companions.
        matches = self._matches(
            ("variable_definition", 0.562),
            ("variable_composition", 0.561),
            ("literal_assignments", 0.559),
        )

        self.assertEqual(confident_aspects(matches), ())

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

    def test_a_capability_names_every_command_family_it_holds(self) -> None:
        # Queue commands were absent from the CICS description, so asking whether
        # the program writes to a queue drifted to the nearest storage-shaped
        # capability and reported an absence that was true of JCL, not of queues.
        cics = CAPABILITY_DESCRIPTORS["cics_evidence"].lower()
        for command_family in ("queue", "send", "receive", "link", "abend"):
            with self.subTest(family=command_family):
                self.assertIn(command_family, cics)

    def test_outbound_lineage_describes_where_a_value_arrives(self) -> None:
        # The aspect has to win on "reaches", "ends up" and "feeds into", which
        # is how an outbound question is actually phrased.
        lineage = VARIABLE_ASPECT_DESCRIPTORS["variable_lineage"].lower()
        self.assertIn("reach", lineage)
        self.assertIn("downstream", lineage)
        self.assertNotIn("declared", lineage)

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


class CollectionFollowupTest(unittest.TestCase):
    """A follow-up asks for the set that was just counted, in full."""

    @staticmethod
    def _counted(intent: str = "variable_inventory") -> SessionState:
        return SessionState(
            current_program="PDCBVC", current_intent=intent,
            current_plan={
                "intent": intent, "result_offset": 0,
                "response_contract": {"format": "count"},
            },
        )

    def _plan(self, question: str, state: SessionState) -> QueryPlan:
        scope = QueryScope(program="PDCBVC", programs=("PDCBVC",), intent="general")
        return build_query_plan(question, scope, intent="general", state=state)

    def test_a_bare_imperative_continues_the_topic(self) -> None:
        # Observed production failure: "how many variables are in PDCBVC?" -> 168,
        # then "list them" -> a list of 12 copybooks. The planner's follow-up test
        # required a wh-word, so a bare imperative never inherited the topic and
        # an unrelated capability was free to win.
        for question in ("list them", "list them all", "show them", "give me all of them"):
            with self.subTest(question=question):
                plan = self._plan(question, self._counted())
                self.assertEqual(plan.intent, "variable_inventory")

    def test_the_follow_up_to_a_count_is_exhaustive(self) -> None:
        # Inheriting the intent alone is not enough: the variable renderer caps a
        # non-exhaustive list at 25, so "list them" would print 25 rows under a
        # heading reading 168 -- a truncation with nothing to show it happened.
        plan = self._plan("list them", self._counted())
        self.assertEqual(plan.result_scope, "all")
        # The count contract itself must not carry over; the follow-up asks for
        # the members, not the number again.
        self.assertNotEqual(plan.response_contract.format, "count")

    def test_an_explicitly_bounded_follow_up_stays_bounded(self) -> None:
        plan = self._plan("show me a few of them", self._counted())
        self.assertEqual(plan.intent, "variable_inventory")
        self.assertNotEqual(plan.result_scope, "all")

    def test_naming_something_starts_a_new_subject(self) -> None:
        # The guard that stops session state following the user to another
        # program: a message that names an identifier is not a continuation.
        for question in (
            "List the variables in PDRTWA2",
            "Show them for PDXXXXX",
            "Where is NPAGT written?",
        ):
            with self.subTest(question=question):
                self.assertFalse(_is_explicit_followup(question.lower(), question))

    def test_a_topic_change_does_not_inherit_the_previous_capability(self) -> None:
        for question, forbidden in (
            ("Now give me an overall summary of PDCBVC.", "variable_inventory"),
            ("Which copybooks are unused?", "variable_inventory"),
            ("List the copybooks.", "variable_inventory"),
            ("hi", "variable_inventory"),
        ):
            with self.subTest(question=question):
                self.assertNotEqual(self._plan(question, self._counted()).intent, forbidden)

    def test_scope_and_planner_agree_on_what_a_follow_up_is(self) -> None:
        # The two disagreed, which is why the bug existed: scope.py treated
        # "list them" as a follow-up and the planner did not.
        for question in ("list them", "show them", "and where is it tested?"):
            with self.subTest(question=question):
                self.assertTrue(refers_to_previous_turn(question.lower()))
                self.assertTrue(_is_explicit_followup(question.lower(), question))


class MapEntityTest(unittest.TestCase):
    """A BMS map name is an identifier a user can ask about, like a variable."""

    _OPERATIONS = {
        "program": "PDCBVC",
        "content": {"operations": [
            {"command": "SEND", "paragraph": "SEND-PDCBVC1", "source_file": "PDCBVC.CBL",
             "line_start": 811, "line_end": 812,
             "statement": "EXEC CICS SEND MAP('PDCBVC1') MAPSET('PDCBVCM') ERASE END-EXEC."},
            {"command": "RECEIVE", "paragraph": "RECEIVE-PDCBVC1", "source_file": "PDCBVC.CBL",
             "line_start": 829, "line_end": 829,
             "statement": "EXEC CICS RECEIVE MAP('PDCBVC1') MAPSET('PDCBVCM') END-EXEC."},
            {"command": "SEND", "paragraph": "SEND-OTHER", "source_file": "PDCBVC.CBL",
             "line_start": 900, "line_end": 900,
             "statement": "EXEC CICS SEND MAP('OTHERMAP') MAPSET('OTHERSET') END-EXEC."},
            {"command": "LINK", "paragraph": "LINK-PD1FS00", "source_file": "PDCBVC.CBL",
             "line_start": 500, "line_end": 501,
             "statement": "EXEC CICS LINK PROGRAM('PD1FS00') COMMAREA(WPD1FS00) END-EXEC."},
        ]},
    }

    def _root(self, directory: str) -> Path:
        root = Path(directory)
        (root / "architecture.cics_operations.json").write_text(
            json.dumps(self._OPERATIONS), encoding="utf-8",
        )
        return root

    def test_map_and_mapset_names_are_read_out_of_a_cics_statement(self) -> None:
        self.assertEqual(
            _cics_statement_resources("EXEC CICS SEND MAP('PDCBVC1') MAPSET('PDCBVCM') ERASE END-EXEC."),
            {"PDCBVC1", "PDCBVCM"},
        )
        # MAPSET must not be read as MAP with a stray suffix, and a statement
        # naming no BMS resource contributes nothing.
        self.assertEqual(
            _cics_statement_resources("EXEC CICS LINK PROGRAM('PD1FS00') END-EXEC."), set(),
        )

    def test_a_named_map_narrows_the_operations_to_that_map(self) -> None:
        # Observed production failure: "Which paragraph sends the PDCBVC1 map?"
        # was answered "PDCBVC1 is not present in the analyzed corpus" while the
        # SEND naming it sat in the evidence. Resolving the name is only half of
        # it -- returning every SEND in the program would still not be the answer.
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            plan = QueryPlan(
                intent="cics_operations", program="PDCBVC", programs=("PDCBVC",),
                entities=(EntityReference("PDCBVC", "map", "PDCBVC1", "PDCBVC|MAP|PDCBVC1"),),
            )
            answer = _answer_cics_operations(root, "PDCBVC", plan)
            self.assertIn("SEND-PDCBVC1", answer)
            self.assertIn("RECEIVE-PDCBVC1", answer)
            self.assertNotIn("SEND-OTHER", answer)
            self.assertNotIn("LINK-PD1FS00", answer)

    def test_a_map_with_no_operation_is_reported_rather_than_answered_broadly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            plan = QueryPlan(
                intent="cics_operations", program="PDCBVC", programs=("PDCBVC",),
                entities=(EntityReference("PDCBVC", "map", "NOSUCHMAP", "PDCBVC|MAP|NOSUCHMAP"),),
            )
            answer = _answer_cics_operations(root, "PDCBVC", plan)
            self.assertIn("NOSUCHMAP", answer)
            self.assertNotIn("SEND-PDCBVC1", answer)

    def test_a_mapset_question_routes_to_cics_evidence_not_file_io(self) -> None:
        # "mapset" sat in the datasets_tables keyword branch, so a question about
        # a BMS screen resource was answered from JCL/DB2 evidence.
        for question in ("Which mapset contains PDCBVC1?", "Show me the mapsets used by PDCBVC"):
            with self.subTest(question=question):
                intent, _ = detect_intent_with_basis(question)
                self.assertEqual(intent, "cics_operations")


class IntentAuthorityTest(unittest.TestCase):
    """A keyword match is a hint; only a specific one may overrule the planner."""

    def test_an_intent_resting_on_a_common_noun_is_reported_as_topical(self) -> None:
        # Observed production failures. "name me cobol files that you have access
        # to" and "PDCBVC doesn't write to any queue, does it?" were answered from
        # JCL dataset evidence because the words "files" and "queue" produced a
        # datasets_tables intent at authority confidence, which vetoed a planner
        # that had correctly asked for an artifact inventory and CICS operations.
        for question in (
            "name me cobol files that you have access to",
            "i didnt say JCL i meant .cbl files",
            "PDCBVC doesn't write to any queue, does it?",
        ):
            with self.subTest(question=question):
                _, basis = detect_intent_with_basis(question)
                self.assertEqual(basis, "topical")

    def test_an_intent_resting_on_domain_vocabulary_keeps_its_authority(self) -> None:
        for question in (
            "Find all CICS SEND and RECEIVE commands in PDCBVC.",
            "What datasets does PDCBVC produce?",
            "Which DB2 tables does PDCBVC access?",
            "Show me the mapsets used by PDCBVC",
            "What business rules are implemented in PDCBVC?",
            "Which copybooks are unused in PDCBVC?",
            "What hardcoded literals are in PDCBVC?",
            "Which external programs does PDCBVC call with parameters?",
            "Walk me through the control flow from start to termination",
        ):
            with self.subTest(question=question):
                _, basis = detect_intent_with_basis(question)
                self.assertEqual(basis, "explicit")

    def test_a_topical_intent_stays_below_the_planner_veto_threshold(self) -> None:
        # merge_semantic_plan only overrules the planner at confidence >= 0.9.
        question = "PDCBVC doesn't write to any queue, does it?"
        intent, basis = detect_intent_with_basis(question)
        scope = resolve_query_scope(question, intent=intent, target_program="PDCBVC")
        plan = build_query_plan(question, scope, intent=scope.intent, intent_basis=basis)
        self.assertLess(plan.confidence, 0.9)
        self.assertLess(plan.authority_confidence, 0.9)

    def test_an_explicit_intent_keeps_full_confidence(self) -> None:
        question = "Find all CICS SEND and RECEIVE commands in PDCBVC."
        intent, basis = detect_intent_with_basis(question)
        scope = resolve_query_scope(question, intent=intent, target_program="PDCBVC")
        plan = build_query_plan(question, scope, intent=scope.intent, intent_basis=basis)
        self.assertGreaterEqual(plan.confidence, 0.9)


class SourceAddressDetectionTest(unittest.TestCase):
    """A number after "line" is an address; the word alone is not."""

    def test_single_and_range_addresses_are_read(self) -> None:
        self.assertEqual(source_address_in("What is on line 227 of PDCBVC?")["line_start"], 227)
        span = source_address_in("Show me lines 395 to 401 of PDCBVC.")
        self.assertEqual((span["line_start"], span["line_end"]), (395, 401))
        for phrasing in ("lines 10-20", "lines 10 through 20", "lines 10 thru 20", "lines 10 until 20"):
            with self.subTest(phrasing=phrasing):
                got = source_address_in(phrasing)
                self.assertEqual((got["line_start"], got["line_end"]), (10, 20))

    def test_a_reversed_range_is_normalised(self) -> None:
        got = source_address_in("show lines 401 to 395")
        self.assertEqual((got["line_start"], got["line_end"]), (395, 401))

    def test_the_source_line_field_is_not_an_address(self) -> None:
        # These ask for line numbers to be *included*, not for a line to be read.
        # Treating them as addresses would hijack ordinary evidence questions.
        for question in (
            "Where is NPAGT modified? Include every paragraph and source line.",
            "List the COPY statements with the copybook name and source line.",
            "Show the source lines for each call.",
        ):
            with self.subTest(question=question):
                self.assertIsNone(source_address_in(question))

    def test_a_named_member_and_a_context_window_are_carried(self) -> None:
        got = source_address_in("Show line 9 of PDSAVTW2.CPY")
        self.assertEqual(got["source_file"], "PDSAVTW2.CPY")
        window = source_address_in("Show 3 lines around line 397")
        self.assertEqual(window["context_before"], 3)
        self.assertEqual(window["context_after"], 3)
        self.assertEqual(window["line_start"], 397)

    def test_every_address_in_the_question_is_read(self) -> None:
        # Observed production failure: "What is on line 227 and 229 of PDCBVC?"
        # answered line 227 only. The reader stopped at the first match, so a
        # truncated answer was indistinguishable from a complete one.
        for question, expected in (
            ("What is on line 227 and 229 of PDCBVC?", [(227, 227), (229, 229)]),
            ("what is on lines 10-20 and 30 of PDCBVC?", [(10, 20), (30, 30)]),
            ("show lines 5, 7 and 9", [(5, 5), (7, 7), (9, 9)]),
            ("Show me lines 1-12 of PDCBVC", [(1, 12)]),
            ("lines 100 to 110", [(100, 110)]),
        ):
            with self.subTest(question=question):
                got = [
                    (a["line_start"], a["line_end"])
                    for a in source_addresses_in(question)
                ]
                self.assertEqual(got, expected)

    def test_a_second_address_does_not_appear_out_of_ordinary_wording(self) -> None:
        # The list reader must not turn evidence questions into address lookups,
        # and a context window states a window rather than a second address.
        for question in (
            "Where is NPAGT modified? Include every paragraph and source line.",
            "How many lines does PDCBVC have?",
            "Which copybooks are unused?",
        ):
            with self.subTest(question=question):
                self.assertEqual(source_addresses_in(question), ())
        window = source_addresses_in("show me line 300 with 3 lines of context")
        self.assertEqual([(a["line_start"], a["line_end"]) for a in window], [(300, 300)])
        self.assertEqual(window[0]["context_before"], 3)

    def test_an_address_becomes_a_typed_entity(self) -> None:
        entity = source_address_entity("PDCBVC", source_address_in("lines 10-20"))
        self.assertEqual(entity.entity_type, "source_address")
        self.assertEqual(entity.value, "10-20")
        self.assertEqual(entity.entity_key, "PDCBVC|SOURCE_ADDRESS|10-20")


class SourceAddressLookupTest(unittest.TestCase):
    """Reading an address must be a lookup, never a similarity match."""

    @staticmethod
    def _corpus(directory: Path, program: str = "PDCBVC") -> Path:
        root = directory / program
        root.mkdir(parents=True)
        rows = [
            {"program": program, "source_file": f"{program}.CBL", "line": n,
             "text": f"      line {n} text", "normalized": f"line {n} text",
             "indicator": " ", "is_comment": False, "is_continuation": False,
             "is_blank": False, "division": "PROCEDURE DIVISION", "section": None,
             "paragraph": "MAIN", "sha256": "0" * 16}
            for n in range(1, 6)
        ]
        rows.append({"program": program, "source_file": "MEMBER.CPY", "line": 1,
                     "text": "      copybook line", "normalized": "copybook line",
                     "indicator": " ", "is_comment": False, "is_continuation": False,
                     "is_blank": False, "division": None, "section": None,
                     "paragraph": None, "sha256": "0" * 16})
        (root / "program.source_lines.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
        )
        return root

    def _answer(self, directory: str, *args, **kwargs) -> str | None:
        root = Path(directory)
        with (
            patch("cobol_rag.final_scripts_answers.find_final_scripts_root", return_value=root),
            patch("cobol_rag.final_scripts_answers.find_program_artifact_root",
                  side_effect=lambda r, p: Path(r) / p),
        ):
            return answer_source_lines(*args, **kwargs)

    def _spans(self, directory: str, program: str, addresses) -> str | None:
        root = Path(directory)
        with (
            patch("cobol_rag.final_scripts_answers.find_final_scripts_root", return_value=root),
            patch("cobol_rag.final_scripts_answers.find_program_artifact_root",
                  side_effect=lambda r, p: Path(r) / p),
        ):
            return answer_source_line_spans(program, addresses)

    def test_two_addresses_are_both_answered_and_accounted_for(self) -> None:
        # Observed production failure: "line 227 and 229" answered 227 only, with
        # nothing in the reply to show a second address had been asked for.
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(Path(directory))
            answer = self._spans(directory, "PDCBVC", [
                {"line_start": 2, "line_end": 2},
                {"line_start": 4, "line_end": 4},
            ])
            self.assertIn("line 2 text", answer)
            self.assertIn("line 4 text", answer)
            self.assertNotIn("line 3 text", answer)
            self.assertIn("2/2 requested address(es)", answer)
            self.assertIn("Returned 2 physical line(s)", answer)

    def test_an_unreachable_address_is_reported_rather_than_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(Path(directory))
            answer = self._spans(directory, "PDCBVC", [
                {"line_start": 2, "line_end": 2},
                {"line_start": 900, "line_end": 900},
            ])
            self.assertIn("line 2 text", answer)
            # The out-of-range address still reports the real bound rather than
            # disappearing from the answer.
            self.assertIn("does not exist", answer)

    def test_the_inventory_names_the_cobol_source_members(self) -> None:
        # "name me cobol files that you have access to" was answered with a list
        # of analysis JSON only, which never mentions a .CBL or .CPY member.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._corpus(root)
            answer = _answer_artifact_inventory(root / "PDCBVC", "PDCBVC")
            self.assertIn("PDCBVC.CBL", answer)
            self.assertIn("MEMBER.CPY", answer)
            self.assertIn("physical line(s)", answer)

    def test_the_exact_line_is_returned_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(Path(directory))
            answer = self._answer(directory, "PDCBVC", 3)
            self.assertIn("      line 3 text", answer)
            self.assertIn("PROCEDURE DIVISION", answer)
            self.assertIn("no line was inferred", answer)

    def test_a_range_returns_every_line_in_the_span(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(Path(directory))
            answer = self._answer(directory, "PDCBVC", 2, 4)
            for n in (2, 3, 4):
                self.assertIn(f"line {n} text", answer)
            self.assertNotIn("line 5 text", answer)

    def test_a_line_beyond_the_file_is_refused_with_the_real_bound(self) -> None:
        # Inventing a plausible line is the worst failure this capability could
        # have, so an out-of-range address reports the range that exists.
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(Path(directory))
            answer = self._answer(directory, "PDCBVC", 900)
            self.assertIn("5 physical line(s)", answer)
            self.assertIn("does not exist", answer)

    def test_context_is_marked_apart_from_the_requested_span(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(Path(directory))
            answer = self._answer(directory, "PDCBVC", 3, context_before=1, context_after=1)
            requested = [l for l in answer.splitlines() if l.startswith("  ") and "line 3 text" in l]
            context = [l for l in answer.splitlines() if l.startswith("\u00b7")]
            self.assertTrue(requested)
            self.assertEqual(len(context), 2)

    def test_a_named_member_selects_that_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(Path(directory))
            answer = self._answer(directory, "PDCBVC", 1, source_file="MEMBER.CPY")
            self.assertIn("copybook line", answer)
            self.assertNotIn("line 1 text", answer)

    def test_an_unknown_member_lists_what_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._corpus(Path(directory))
            answer = self._answer(directory, "PDCBVC", 1, source_file="NOPE.CPY")
            self.assertIn("not a recorded source member", answer)
            self.assertIn("MEMBER.CPY", answer)

    def test_a_program_with_no_source_map_declines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "PDCBVC").mkdir()
            self.assertIsNone(self._answer(directory, "PDCBVC", 1))


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


class TechnicalAnswerabilityGateTest(unittest.TestCase):
    def test_semantically_incoherent_relation_is_rejected(self) -> None:
        llm = Mock()
        llm.complete.return_value = Mock(text=json.dumps({
            "answerable": False,
            "reason": "color/time relation has no COBOL evidence meaning",
            "capability": "",
        }))
        with patch("cobol_rag.query.build_llm", return_value=llm):
            answerable, reason = _assess_technical_answerability(
                "Which paragraph is the blue value of PDCBVC before yesterday?",
                AppConfig(), QueryScope(program="PDCBVC"),
            )
        self.assertFalse(answerable)
        self.assertIn("no COBOL evidence meaning", reason)

    def test_coherent_broad_cics_request_is_accepted(self) -> None:
        llm = Mock()
        llm.complete.return_value = Mock(text=json.dumps({
            "answerable": True,
            "reason": "maps to CICS operations",
            "capability": "cics_evidence",
        }))
        with patch("cobol_rag.query.build_llm", return_value=llm):
            answerable, _ = _assess_technical_answerability(
                "Find the CICS commands executed by PDCBVC.",
                AppConfig(), QueryScope(program="PDCBVC"),
            )
        self.assertTrue(answerable)

    def test_typed_retrieval_includes_normalized_evidence_view(self) -> None:
        plan = QueryPlan(
            route="technical", intent="variable_dataflow", domain="dataflow",
            tasks=("variable_reads",),
            source_domains=("dataflow.variable",),
        )
        self.assertIn("evidence.normalized", _chunk_types_for_plan(plan))

    def test_natural_external_call_inventory_wording_has_a_typed_task(self) -> None:
        plan = build_query_plan(
            "Which programs does PDCBVC call?",
            QueryScope(program="PDCBVC", programs=("PDCBVC",)),
        )
        self.assertEqual(plan.intent, "external_programs")
        self.assertEqual(plan.tasks, ("external_calls",))

    def test_exact_line_qualifier_does_not_hide_program_summary_intent(self) -> None:
        plan = build_query_plan(
            "Explain PDCBVC in exactly three lines.",
            QueryScope(program="PDCBVC", programs=("PDCBVC",)),
        )
        self.assertEqual(plan.intent, "program_summary")
        self.assertEqual(plan.tasks, ("program_summary",))
        self.assertEqual(plan.response_contract.exact_lines, 3)


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

    def test_a_confidently_ranked_capability_recompiles_a_weak_plan(self) -> None:
        # A weak semantic plan and an independent confident capability vote must
        # not become two unrelated claims. The compiler corrects the plan to the
        # evidence capability that can answer the request.
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent="control_flow", domain="control_flow",
            subtasks=(EvidenceSubtask(
                claim_id="claim_1", description="Verify control flow evidence",
                capability="control_flow", tasks=("complete_program_flow",),
            ),),
        )

        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("cics_evidence", 0.77, 0.09),),
        ):
            supplemented = _supplement_missing_capability(
                "Find all CICS SEND and RECEIVE commands in PDCBVC.",
                Mock(), QueryScope(intent="control_flow", program="PDCBVC"), plan,
            )

        self.assertEqual(
            [s.capability for s in supplemented.subtasks], ["cics_evidence"],
        )
        self.assertEqual(supplemented.intent, "cics_operations")
        self.assertIn("semantic_claims_recompiled", supplemented.policy_rejections[0])

    def test_capability_compiler_removes_relations_the_selected_handler_cannot_execute(self) -> None:
        call = EntityReference("PDCBVC", "call", "PXRSEMAF", "PDCBVC|PXRSEMAF|CALL")
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent="external_programs", domain="integration", authority_confidence=0.7,
            tasks=("call_context",), entities=(call,),
            relations=("before", "after", "writes", "contains", "starts_at"),
            subtasks=(EvidenceSubtask(
                claim_id="claim_1", description="confused semantic claim",
                capability="variable_access", tasks=("variable_writes",),
                entity_values=("PXRSEMAF",),
            ),),
        )
        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("call_context", 0.82, 0.12),),
        ):
            result = _supplement_missing_capability(
                "Which fields are filled in ahead of the PXRSEMAF call?",
                Mock(), QueryScope(program="PDCBVC", entities=(call,)), plan,
            )
        self.assertEqual(result.relations, ("before",))
        self.assertTrue(_direct_handler_supports(result))

    def test_semantic_confidence_cannot_promote_itself_to_typed_authority(self) -> None:
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent="control_flow", domain="control_flow",
            confidence=0.99, authority_confidence=0.45,
            subtasks=(EvidenceSubtask(
                claim_id="claim_1", description="semantic control-flow guess",
                capability="control_flow", tasks=("complete_program_flow",),
            ),),
        )
        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("cics_evidence", 0.77, 0.09),),
        ):
            result = _supplement_missing_capability(
                "Find all CICS commands.", Mock(), QueryScope(program="PDCBVC"), plan,
            )
        self.assertEqual(result.subtasks[0].capability, "cics_evidence")

    def test_agreement_between_planner_and_ranking_changes_nothing(self) -> None:
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent="cics_operations", domain="integration",
            subtasks=(EvidenceSubtask(
                claim_id="claim_1", description="Verify cics evidence",
                capability="cics_evidence", tasks=("cics_operations",),
            ),),
        )

        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("cics_evidence", 0.77, 0.09),),
        ):
            self.assertEqual(
                _supplement_missing_capability("q", Mock(), None, plan).subtasks, plan.subtasks,
            )

    def test_program_wide_similarity_does_not_pollute_named_variable_claim(self) -> None:
        variable = EntityReference(
            program="PDCBVC", entity_type="variable", value="WDATE2-GG",
            entity_key="PDCBVC|VARIABLE|WDATE2-GG",
        )
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent="variable_dataflow", domain="dataflow",
            tasks=("variable_definition",), entities=(variable,),
            subtasks=(EvidenceSubtask(
                claim_id="claim_1", description="Verify variable access",
                capability="variable_access", tasks=("variable_definition",),
                entity_values=("WDATE2-GG",),
            ),),
        )
        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("db2_evidence", 0.77, 0.09),),
        ):
            result = _supplement_missing_capability(
                "Include its declaration and parent group.", Mock(),
                QueryScope(program="PDCBVC", entities=(variable,)), plan,
            )
        self.assertEqual(result.subtasks, plan.subtasks)

    def test_variable_refinement_preserves_explicit_declaration_task(self) -> None:
        variable = EntityReference(
            program="PDCBVC", entity_type="variable", value="WDATE2-GG",
            entity_key="PDCBVC|VARIABLE|WDATE2-GG",
        )
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent="variable_dataflow", domain="dataflow",
            tasks=("variable_definition",), entities=(variable,),
        )
        with (
            patch("cobol_rag.query.router_for", return_value=Mock(rank_aspects=lambda q: ())),
            patch("cobol_rag.query.confident_aspects", return_value=()),
            patch("cobol_rag.query._resolve_dataflow_direction", return_value="reads"),
        ):
            result = _refine_variable_tasks(
                "What is WDATE2-GG? Include its declaration and parent group.", Mock(), plan,
            )
        self.assertIn("variable_definition", result.tasks)
        self.assertIn("variable_reads", result.tasks)

    def test_variable_refinement_respects_an_explicit_read_only_followup(self) -> None:
        variable = EntityReference(
            program="PDCBVC", entity_type="variable", value="NPAGT",
            entity_key="PDCBVC|VARIABLE|NPAGT",
        )
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent="variable_dataflow", domain="dataflow",
            tasks=("variable_reads",), entities=(variable,),
        )
        with (
            patch("cobol_rag.query.router_for", return_value=Mock(rank_aspects=lambda q: ())),
            patch("cobol_rag.query.confident_aspects", return_value=("variable_writes",)),
            patch("cobol_rag.query._resolve_dataflow_direction", return_value="both"),
        ):
            result = _refine_variable_tasks("And where is it tested?", Mock(), plan)
        self.assertIn("variable_reads", result.tasks)
        self.assertNotIn("variable_writes", result.tasks)

    def test_an_unconfident_ranking_adds_nothing(self) -> None:
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent="control_flow", domain="control_flow",
            subtasks=(EvidenceSubtask(
                claim_id="claim_1", description="Verify control flow evidence",
                capability="control_flow", tasks=("complete_program_flow",),
            ),),
        )

        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("cics_evidence", 0.48, 0.01),),
        ):
            self.assertEqual(
                _supplement_missing_capability("q", Mock(), None, plan).subtasks, plan.subtasks,
            )

    def test_a_capability_needing_an_identifier_is_not_added_without_one(self) -> None:
        # variable_access has nothing to say without a resolved variable, so a
        # confident ranking cannot introduce an empty claim.
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), route="technical",
            intent="program_summary", domain="program_structure",
            subtasks=(EvidenceSubtask(
                claim_id="claim_1", description="Verify program summary",
                capability="program_summary", tasks=("program_summary",),
            ),),
        )

        with patch(
            "cobol_rag.query._rank_question_capabilities",
            return_value=(CapabilityMatch("variable_access", 0.80, 0.10),),
        ):
            self.assertEqual(
                _supplement_missing_capability("q", Mock(), None, plan).subtasks, plan.subtasks,
            )

    def test_a_program_wide_claim_reroutes_on_a_confident_disagreement(self) -> None:
        # Asking for the CICS commands ranks unambiguously as CICS evidence while
        # the planner reaches for control flow. With no entity to narrow by, the
        # ranking arbitrates only when it is confident.
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="control_flow",
            domain="control_flow", tasks=("complete_program_flow",),
        )
        subtask = replace(self._subtask("control_flow"), entity_values=())

        with (
            patch("cobol_rag.query.unavailable_capabilities", return_value=frozenset()),
            patch(
                "cobol_rag.query.router_for",
                return_value=Mock(rank=lambda q, allowed=None: (CapabilityMatch("cics_evidence", 0.77, 0.09),)),
            ),
        ):
            rerouted = _rerouted_subtask(
                question="Find all CICS SEND and RECEIVE commands in PDCBVC.",
                config=Mock(), plan=plan, subtask=subtask,
                tried=frozenset({"control_flow"}),
            )

        self.assertIsNotNone(rerouted)
        self.assertEqual(rerouted.capability, "cics_evidence")

    def test_an_unconfident_ranking_never_overrides_the_planner(self) -> None:
        plan = QueryPlan(
            program="PDCBVC", programs=("PDCBVC",), intent="control_flow",
            domain="control_flow", tasks=("complete_program_flow",),
        )
        subtask = replace(self._subtask("control_flow"), entity_values=())

        with (
            patch("cobol_rag.query.unavailable_capabilities", return_value=frozenset()),
            patch(
                "cobol_rag.query.router_for",
                return_value=Mock(rank=lambda q, allowed=None: (CapabilityMatch("cics_evidence", 0.48, 0.01),)),
            ),
        ):
            self.assertIsNone(_rerouted_subtask(
                question="Tell me about this program.",
                config=Mock(), plan=plan, subtask=subtask,
                tried=frozenset({"control_flow"}),
            ))

    def test_any_unresolved_required_claim_is_retried(self) -> None:
        # An off-entity answer is the clearest case, but any required claim the
        # capability could not answer leaves a gap another capability may fill.
        plan = self._plan()
        subtask = self._subtask("condition_outcome")
        for reasons in (
            ("missing_requested_entity:NPAGT",),
            ("unsupported_claim_line:1",),
            ("missing_requested_section:control_outcome",),
        ):
            with self.subTest(reasons=reasons):
                self.assertTrue(_claim_should_be_rerouted(EvidenceSubtaskResult(
                    subtask=subtask, plan=plan, passed=False, reasons=reasons,
                )))

    def test_a_passing_or_optional_claim_is_never_retried(self) -> None:
        plan = self._plan()
        passing = EvidenceSubtaskResult(
            subtask=self._subtask("variable_access"), plan=plan, passed=True, answer="grounded",
        )
        optional = EvidenceSubtaskResult(
            subtask=replace(self._subtask("condition_outcome"), required=False),
            plan=plan, passed=False, reasons=("unsupported_claim_line:1",),
        )

        self.assertFalse(_claim_should_be_rerouted(passing))
        self.assertFalse(_claim_should_be_rerouted(optional))

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
