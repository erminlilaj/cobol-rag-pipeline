from __future__ import annotations

import unittest
from unittest.mock import patch

from cobol_rag.final_scripts_answers import answer_copybook_role
from cobol_rag.query_ir import CopybookRole, compile_query


INCLUSIONS = [
    {"copybook": "LOCALDATA", "division": "DATA DIVISION",
     "section": "WORKING-STORAGE SECTION", "line": 100, "statement": "COPY LOCALDATA."},
    {"copybook": "IFACE", "division": "DATA DIVISION",
     "section": "WORKING-STORAGE SECTION", "line": 110, "statement": "COPY IFACE."},
    {"copybook": "PASSEDIN", "division": "DATA DIVISION",
     "section": "LINKAGE SECTION", "line": 200, "statement": "COPY PASSEDIN."},
    {"copybook": "CODEBIT", "division": "PROCEDURE DIVISION",
     "section": "UNKNOWN", "line": 800, "statement": "COPY CODEBIT."},
]
CALLS = {"calls": [{"target": "SUBPGM", "call_type": "CICSLINK", "paragraph": "DO-LINK",
                    "line_start": 400, "commarea": "WIFACE",
                    "parameters": ["WIFACE"], "parameter_details": [{"field_prefix": "IFACE"}]}]}
LINEAGE = {"content": {"fields": [{"field": "SCRNO", "origin": "COPY:LOCALDATA"}]}}


class CopybookRoleTest(unittest.TestCase):
    """A copybook's purpose is visible in where it is copied and what the
    program does with its fields. Reading it from the name works only for
    programs that follow the naming standard, and fails silently for the rest.
    """

    def role(self, name, lineage=None):
        with patch("cobol_rag.final_scripts_answers._copybook_inclusions", return_value=INCLUSIONS), \
             patch("cobol_rag.final_scripts_answers.find_final_scripts_root", return_value=__import__("pathlib").Path("/x")), \
             patch("cobol_rag.final_scripts_answers.find_program_artifact_root", return_value=__import__("pathlib").Path("/x/P")), \
             patch("cobol_rag.final_scripts_answers._read_json", return_value=CALLS), \
             patch("cobol_rag.final_scripts_answers._screen_lineage", return_value=lineage):
            return answer_copybook_role("TESTPROG", name) or ""

    def test_copy_in_procedure_division_is_code(self) -> None:
        self.assertIn("executable code", self.role("CODEBIT"))

    def test_copy_in_linkage_is_passed_in(self) -> None:
        self.assertIn("passed in by whatever started", self.role("PASSEDIN"))

    def test_copy_in_working_storage_is_a_local_data_area(self) -> None:
        self.assertIn("data area the program declares for itself", self.role("LOCALDATA"))

    def test_an_area_named_on_a_call_is_an_interface(self) -> None:
        answer = self.role("IFACE")
        self.assertIn("interface used to talk to another program", answer)
        self.assertIn("SUBPGM", answer)

    def test_a_copybook_declaring_screen_fields_is_the_map(self) -> None:
        self.assertIn("screen map", self.role("LOCALDATA", lineage=LINEAGE))

    def test_a_copybook_the_program_does_not_include_has_no_role(self) -> None:
        self.assertEqual(self.role("NOTHERE"), "")

    def test_the_role_never_depends_on_the_name(self) -> None:
        """The same structure under a name following no convention at all."""
        odd = [dict(item, copybook="ZZ9") for item in INCLUSIONS if item["copybook"] == "CODEBIT"]
        with patch("cobol_rag.final_scripts_answers._copybook_inclusions", return_value=odd), \
             patch("cobol_rag.final_scripts_answers.find_final_scripts_root", return_value=__import__("pathlib").Path("/x")), \
             patch("cobol_rag.final_scripts_answers.find_program_artifact_root", return_value=__import__("pathlib").Path("/x/P")), \
             patch("cobol_rag.final_scripts_answers._read_json", return_value=CALLS), \
             patch("cobol_rag.final_scripts_answers._screen_lineage", return_value=None):
            self.assertIn("executable code", answer_copybook_role("TESTPROG", "ZZ9") or "")


class CopybookPurposeCompileTest(unittest.TestCase):
    def compile(self, question):
        return compile_query(question, program="TESTPROG", copybooks=("PDRTWA2",), graph_nodes=())

    def test_purpose_questions_compile_to_a_role_query(self) -> None:
        for phrasing in (
            "what is PDRTWA2 for?",
            "what is PDRTWA2 used for?",
            "why is PDRTWA2 copied here?",
            "tell me about PDRTWA2",
        ):
            with self.subTest(phrasing=phrasing):
                self.assertIsInstance(self.compile(phrasing), CopybookRole)

    def test_a_listing_question_is_not_a_role_query(self) -> None:
        self.assertNotIsInstance(self.compile("which copybooks does it use?"), CopybookRole)


if __name__ == "__main__":
    unittest.main()
