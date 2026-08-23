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

    def test_two_games_are_available_in_the_waiting_component(self) -> None:
        self.assertIn("function createWaitingExperience()", self.script)
        self.assertIn("Bug Hunt", self.script)
        self.assertIn("COBOL Match", self.script)
        self.assertIn("Catch the moving bug.", self.script)
        self.assertIn("Match the COBOL keywords.", self.script)

    def test_waiting_component_is_destroyed_for_success_error_and_final_cleanup(self) -> None:
        send_message = self.script.split("async function sendMessage()", 1)[1]
        self.assertIn("const waiting = createWaitingExperience();", send_message)
        self.assertGreaterEqual(send_message.count("waiting.destroy();"), 3)
        self.assertGreaterEqual(send_message.count("waiting.element.remove();"), 3)

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


if __name__ == "__main__":
    unittest.main()
