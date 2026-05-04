from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cobol_rag.config import load_config  # noqa: E402
from cobol_rag.final_scripts_answers import find_final_scripts_root  # noqa: E402
from cobol_rag.query import answer_query  # noqa: E402


TESTS = [
    (
        "phase_dispatch",
        "In PDCBVC.CBL, how does the program decide whether to execute BROWSE-FASE1 or BROWSE-FASE2, "
        "and what happens if TWCOB-FASE contains an unexpected value?",
        ["TWCOB-FASE = '1'", "BROWSE-FASE1", "TWCOB-FASE = '2'", "BROWSE-FASE2", "ABEND00"],
    ),
    (
        "semaphore_flow",
        "In PDCBVC.CBL, under what exact conditions does the program read the semaphore PDAGGVIP, "
        "and what control-flow path is taken when the semaphore status is closed?",
        ["READ-TAB-SEMAF", "PDAGGVIP", "PXCSEMAF-STATUS = 1", "XCTL-LIV4"],
    ),
    (
        "browse_fase1_sequence",
        "In PDCBVC.CBL, explain the full sequence of operations performed during BROWSE-FASE1 before "
        "the map PDCBVC1 is sent to the terminal.",
        ["PREP-LINK-PD1FS00", "INIZ-PARAM", "LINK-PD1VOCI", "CALCOLA-NPAG", "PREPARA-MAP", "SEND-PDCBVC1"],
    ),
    (
        "pd1voci_preparation",
        "In PDCBVC.CBL, how are the parameters for the external program PD1VOCI prepared, and how do "
        "TWCOB-VARCONT-NUMFUNZ and TWCOB-FUNZIONE influence PD1VOCI-FUNZIONE, PD1VOCI-TIPO-ESTRA, "
        "and PD1VOCI-TIPO-VOCE?",
        ["PD1VOCI-FUNZIONE", "PD1VOCI-TIPO-ESTRA", "PD1VOCI-TIPO-VOCE", "TWCOB-VARCONT-NUMFUNZ"],
    ),
    (
        "pagination",
        "In PDCBVC.CBL, how is pagination calculated and maintained across user interactions, especially "
        "when the user presses ENTER, PF7, or PF8?",
        ["WCTPAG", "TWCOB-VARCONT-NPAGINA", "BROWSE-FASE2-ENTER", "BROWSE-FASE2-PF7", "BROWSE-FASE2-PF8"],
    ),
    (
        "row_selection",
        "In PDCBVC.CBL, when the user selects a row from the browse screen, how does the program validate "
        "the selected progressivo and move the selected accounting voice into the TWA before transferring "
        "control to the next program?",
        ["SCELTAI", "WPROGR", "BROWSE-FASE2-SEL-20", "TWCOB-VARCONT-PROGVOCE"],
    ),
    (
        "pf_key_comparison",
        "In PDCBVC.CBL, compare the control-flow behavior of function keys PF1, PF2, PF3, PF4, and PF9. "
        "Which target programs are selected, and what common TWA reset logic is applied?",
        ["DFHPF1", "XCTL-LIV1", "DFHPF9", "XCTL-LIV0", "RESET-TWA", "PDPRED"],
    ),
    (
        "error_paths",
        "In PDCBVC.CBL, identify all paths that lead to an error message or abnormal termination, including "
        "invalid function keys, missing records, invalid selection, failed service calls, SQL errors, and "
        "semaphore restrictions.",
        ["ABEND00", "BROWSE-FASE2-TASTOER", "BROWSE-FASE2-NOTFND", "SQLERROR", "PXCSEMAF-STATUS"],
    ),
    (
        "generic_pf8_key",
        "What happens when the user presses PF8 in PDCBVC?",
        ["DFHPF8", "BROWSE-FASE2-PF8"],
    ),
    (
        "generic_paragraph",
        "Explain BROWSE-FASE2-ENTER in PDCBVC.",
        ["BROWSE-FASE2-ENTER", "PREP-LINK-PD1FS00", "LINK-PD1VOCI"],
    ),
    (
        "generic_call_target",
        "Which parameters are passed to PD1FS00 in PDCBVC?",
        ["COMMAREA=WPD1FS00", "LENGTH=PD1FS00-LUNGH", "PD1FS00-FUNZIONE"],
    ),
    (
        "generic_variable",
        "Where is TWCOB-VARCONT-NPAGINA set and used?",
        ["TWCOB-VARCONT-NPAGINA", "BROWSE-FASE1", "BROWSE-FASE2-VISUAL"],
    ),
]


def main() -> int:
    if not os.environ.get("COBOL_RAG_FINAL_SCRIPTS_DIR") and find_final_scripts_root() is None:
        print("SKIP: set COBOL_RAG_FINAL_SCRIPTS_DIR or place final_scripts at the repo root.")
        return 0

    config = load_config()
    failures: list[str] = []
    for name, question, fragments in TESTS:
        answer = answer_query(question, config).answer
        missing = [fragment for fragment in fragments if fragment not in answer]
        if missing:
            failures.append(f"{name}: missing {', '.join(missing)}")
            print(f"FAIL {name}")
            print(answer)
        else:
            print(f"PASS {name}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
