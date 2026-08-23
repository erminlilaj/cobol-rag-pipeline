from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from cobol_rag.chat import ChatSession
from cobol_rag.config import load_config
from cobol_rag.query import QueryError


app = FastAPI(title="COBOL RAG UI")
session = ChatSession(config=load_config())


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, object]:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    try:
        answer = await run_in_threadpool(session.ask, message)
    except QueryError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {
        "answer": answer.answer,
        "route": answer.route,
        "scope": answer.scope.as_dict(),
        "plan": answer.plan.as_dict() if answer.plan else {},
        "guard_status": answer.guard_status,
        "trace_id": answer.trace_id,
        "execution_mode": answer.execution_mode,
        "debug": answer.debug,
        "sources": [
            {
                "rank": index,
                "score": round(source.score, 4),
                "chunk_type": source.metadata.get("chunk_type", ""),
                "source_id": source.metadata.get("source_id", ""),
                "preview": source.text[:500],
            }
            for index, source in enumerate(answer.sources, start=1)
        ],
    }


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>COBOL RAG</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101316;
      --panel: #171c20;
      --muted: #9aa7b3;
      --text: #f1f5f7;
      --line: #2a333a;
      --accent: #47c2a8;
      --danger: #ff7979;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 16px;
      padding: 22px 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
    }
    #log {
      overflow: auto;
      padding-right: 4px;
    }
    .turn {
      border-bottom: 1px solid var(--line);
      padding: 18px 0;
    }
    .role {
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 8px;
    }
    .answer {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .sources {
      margin-top: 14px;
      display: grid;
      gap: 8px;
    }
    details {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }
    summary {
      cursor: pointer;
      color: var(--muted);
    }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 10px 0 0;
      color: #dce6ec;
    }
    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    textarea {
      min-height: 54px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }
    button {
      width: 96px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #07110f;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: .55;
      cursor: wait;
    }
    .mode {
      margin: 8px 0;
      color: var(--muted);
      font-size: 12px;
    }
    .error { color: var(--danger); }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>COBOL RAG</h1>
      <div class="status">collection: cobol-dev | llm: granite-code:8b-instruct</div>
    </header>
    <section id="log">
      <div class="turn">
        <div class="role">Ready</div>
        <div class="answer">Ask about PDCBVC. Sources will appear under each answer.</div>
      </div>
    </section>
    <form id="form">
      <textarea id="message" placeholder="Example: What is the purpose of program PDCBVC?" autofocus></textarea>
      <button id="send" type="submit">Ask</button>
    </form>
  </main>
  <script>
    const form = document.querySelector("#form");
    const message = document.querySelector("#message");
    const send = document.querySelector("#send");
    const log = document.querySelector("#log");

    function addTurn(role, html, className = "") {
      const turn = document.createElement("div");
      turn.className = "turn";
      turn.innerHTML = `<div class="role">${role}</div><div class="answer ${className}">${html}</div>`;
      log.appendChild(turn);
      log.scrollTop = log.scrollHeight;
      return turn;
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[char]));
    }

    function renderSources(sources) {
      if (!sources.length) return "";
      return `<div class="sources">${sources.map(source => `
        <details>
          <summary>#${source.rank} ${escapeHtml(source.chunk_type || "source")} ${escapeHtml(source.source_id)}</summary>
          <pre>${escapeHtml(source.preview)}</pre>
        </details>`).join("")}</div>`;
    }

    form.addEventListener("submit", async event => {
      event.preventDefault();
      const text = message.value.trim();
      if (!text) return;
      addTurn("You", escapeHtml(text));
      message.value = "";
      send.disabled = true;
      const pending = addTurn("RAG", "Thinking...");
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Request failed.");
        const mode = data.execution_mode
          ? `<div class="mode">${escapeHtml(data.execution_mode.replaceAll("_", " "))}</div>`
          : "";
        pending.querySelector(".answer").innerHTML =
          mode + escapeHtml(data.answer) + renderSources(data.sources || []);
      } catch (error) {
        pending.querySelector(".answer").classList.add("error");
        pending.querySelector(".answer").textContent = error.message;
      } finally {
        send.disabled = false;
        message.focus();
      }
    });
  </script>
</body>
</html>
"""
