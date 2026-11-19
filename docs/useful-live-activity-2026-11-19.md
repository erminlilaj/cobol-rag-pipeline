# Useful live activity explanations

The pasted live log showed actual files and model durations, but repeated generic
labels obscured the distinction between inference, disk reads and cached lookup
results. This update adds stage explanations, not hidden model reasoning.

## Visible while running

- A current-operation card states the stage, its purpose and file/source scope.
- Each model call explains why it was invoked: request resolution, next action,
  invalid-argument repair, evidence recovery, answer repair, final/partial
  synthesis, translation or independent review. Phase labels follow the actual
  orchestration path, not model-generated descriptions of its intentions.
- Model events name the configured model, program scope, source member names
  actually represented in supplied preview rows, preview row count, input
  character count, context-window setting and remaining call budget. Characters
  are explicitly not a token estimate. This is supplied context, not a claim
  that a physical source file is currently being opened or fully read.
- During a long model call the local clock shows the last reported stage and
  observed wait. It does not fabricate token streaming, ETA or ongoing file reads.
- File reads explain the artifact's role: indexed source text, CICS operation
  context, CFG conditions, call parameters or corpus relationships. Cached
  artifacts and repeated tool results explicitly say no new read/lookup occurred.
- Tool results state what was returned. Outline hits list paragraph locations;
  zero matches and incomplete source windows explain their coverage limitations.
- Repeated rounds with no new facts, bounded recovery reads, draft availability
  and independent review verdicts are visible events.
- Source previews favor real conditions and operations over blanks/header lines.
  They remain verbatim unverified source excerpts, not accepted answers.

No extra model requests, provider settings, analysis, indexing or memory changes
are introduced. The existing session-owned bounded feed and cleanup behavior are
preserved. No prompts, draft prose or private model reasoning are sent in events.

Validation: focused Python investigation/activity/UI/usage/language tests and
JavaScript chat/export tests pass. The full suite retains the same 21 known
legacy planner/router failures. Tests cover bounded metadata, source selection,
typed-stage repair, zero-match explanations, active-model clock clearing,
session isolation and malformed-review logging safety.

Pull feature/batch-question-testing, run start_rag.sh and refresh the browser.
No setup or reanalysis is necessary. A request already running before restart
will not gain the new instrumentation retroactively.
