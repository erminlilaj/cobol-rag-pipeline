"""Every artifact the pipeline writes must have something that reads it.

screen_field_lineage.json was generated on every run, indexed, and read by no
capability at all. A question about a screen field therefore had no path and
was answered by generation, which returned the field's own name as its
description. Nothing detected that, because nothing compared the artifacts
produced against the artifacts consumed.

This test is that comparison. A new artifact with no reader fails here rather
than when a question happens to reach for it.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from cobol_rag.final_scripts_answers import find_final_scripts_root


SOURCE_DIR = Path(__file__).parents[1] / "src" / "cobol_rag"

# Artifacts deliberately not read by a capability. Each needs a reason, so
# that adding one is a decision rather than a way to silence this test.
NOT_READ_BY_NAME = {
    # Read through the catalogue rather than by filename.
    "corpus.registry.json",
    # Sidecar to program.source_lines.jsonl, which is what capabilities read.
    "program.source_lines.meta.json",
    # An analysis-side diagnostic about the artifacts themselves, not evidence
    # about the COBOL. Worth a capability eventually -- "are there analysis
    # discrepancies in this program" is a fair question -- but it is not one
    # today, and that is a choice rather than an oversight.
    "quality.reconciliation_report.json",
}


def artifact_names() -> set[str]:
    root = find_final_scripts_root()
    if root is None:
        return set()
    names: set[str] = set()
    for program_dir in sorted(root.iterdir()):
        if not program_dir.is_dir():
            continue
        for path in program_dir.glob("*.json"):
            names.add(path.name)
    return names


def source_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in SOURCE_DIR.rglob("*.py")
    )


class ArtifactReaderTest(unittest.TestCase):
    def test_every_artifact_has_a_reader(self) -> None:
        names = artifact_names()
        if not names:
            # Set COBOL_RAG_FINAL_SCRIPTS_DIR to the generated corpus to run
            # this; without it the check cannot see what the pipeline produced.
            self.skipTest("no generated corpus available")
        text = source_text()
        orphans = sorted(
            name
            for name in names
            if name not in NOT_READ_BY_NAME
            # A reader names the file, or names the stem it is looked up by.
            and name not in text
            and re.sub(r"\.json$", "", name) not in text
        )
        self.assertEqual(
            orphans,
            [],
            f"artifacts produced but read by no capability: {orphans}",
        )


if __name__ == "__main__":
    unittest.main()
