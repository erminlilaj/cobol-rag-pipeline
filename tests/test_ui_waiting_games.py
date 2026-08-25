from __future__ import annotations

import unittest
from pathlib import Path


class WaitingGamesAssetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.script = (root / "ui" / "app.js").read_text(encoding="utf-8")
        cls.styles = (root / "ui" / "style.css").read_text(encoding="utf-8")
        cls.index = (root / "ui" / "index.html").read_text(encoding="utf-8")
        cls.api = (root / "src" / "cobol_rag" / "api.py").read_text(encoding="utf-8")
        cls.chat = (root / "src" / "cobol_rag" / "chat.py").read_text(encoding="utf-8")

    def test_two_games_are_available_in_the_waiting_component(self) -> None:
        self.assertIn("function createWaitingExperience()", self.script)
        self.assertIn("Bug Hunt", self.script)
        self.assertIn("COBOL Match", self.script)
        self.assertIn("Catch the moving bug.", self.script)
        self.assertIn("Match the COBOL keywords.", self.script)

    def test_waiting_component_is_always_destroyed_however_the_request_ends(self) -> None:
        # The component belongs to one request, so it now lives in askOne. The
        # three duplicated cleanups collapsed into a single finally, which is a
        # stronger guarantee than repetition: no exit path can skip it.
        ask_one = self.script.split("async function askOne(", 1)[1].split("\nasync function", 1)[0]
        self.assertIn("const waiting = createWaitingExperience();", ask_one)
        self.assertIn("} finally {", ask_one)
        cleanup = ask_one.split("} finally {", 1)[1]
        self.assertIn("waiting.destroy();", cleanup)
        self.assertIn("waiting.element.remove();", cleanup)
        self.assertEqual(ask_one.count("waiting.destroy();"), 1)

    def test_a_running_question_can_be_stopped_from_the_composer(self) -> None:
        self.assertIn('id="stop-btn"', self.index)
        self.assertIn('onclick="stopChat()"', self.index)
        self.assertIn("new AbortController()", self.script)
        self.assertIn("signal: runningRequest.signal", self.script)
        self.assertIn("runningRequest.abort()", self.script)
        self.assertIn("'/api/chat/cancel'", self.script)

    def test_stopping_is_reported_and_kept_out_of_chat_memory(self) -> None:
        self.assertIn("AbortError", self.script)
        self.assertIn('@app.post("/api/chat/cancel")', self.api)
        self.assertIn("def cancel(self) -> int:", self.chat)
        # Reset must also stand down whatever is running, or the discarded
        # answer lands in a session the user believes they just cleared.
        chat_reset = self.api.split('@app.post("/api/chat/reset")', 1)[1]
        self.assertIn("session.cancel()", chat_reset)

    def test_multiple_questions_are_queued_and_asked_one_at_a_time(self) -> None:
        self.assertIn('id="queue-bar"', self.index)
        self.assertIn('id="queue-list"', self.index)
        self.assertIn("function splitQuestions(", self.script)
        self.assertIn("async function drainQueue()", self.script)
        drain = self.script.split("async function drainQueue()", 1)[1].split("\nasync function", 1)[0]
        # Sequential by construction: one await per iteration, no Promise.all.
        self.assertIn("while (pendingQuestions.length)", drain)
        self.assertIn("await askOne(next)", drain)
        self.assertNotIn("Promise.all", drain)

    def test_stopping_a_batch_cancels_the_questions_still_queued(self) -> None:
        drain = self.script.split("async function drainQueue()", 1)[1].split("\nasync function", 1)[0]
        self.assertIn("pendingQuestions = []", drain)
        self.assertIn("cancelled", drain)

    def test_games_have_responsive_and_reduced_motion_styles(self) -> None:
        for selector in (
            ".waiting-experience",
            ".waiting-game-tabs",
            ".bug-grid",
            ".memory-grid",
            ".memory-card.matched",
        ):
            self.assertIn(selector, self.styles)
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.styles)

    def test_ui_assets_are_versioned_and_served_without_stale_browser_caching(self) -> None:
        self.assertIn("style.css?v=waiting-games-", self.index)
        self.assertIn("app.js?v=waiting-games-", self.index)
        self.assertIn('response.headers["Cache-Control"] = "no-store', self.api)

    def test_each_browser_has_isolated_chat_memory(self) -> None:
        self.assertIn("cobol-rag-session-id", self.script)
        self.assertIn("X-Session-ID", self.script)
        self.assertIn("_chat_sessions", self.api)
        self.assertIn("_request_session_id(request", self.api)


if __name__ == "__main__":
    unittest.main()
