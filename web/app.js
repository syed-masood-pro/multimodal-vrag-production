"use strict";

// Incremental SSE parser. Keep a trailing CR until the next chunk so CRLF is
// never mistaken for two line endings. Data lines are joined with a newline.
function createSSEParser(onEvent) {
  let buffer = "", event = "message", data = [];
  function line(text) {
    if (!text) {
      if (data.length) onEvent(event, data.join("\n"));
      event = "message"; data = [];
      return;
    }
    if (text.startsWith(":")) return;
    const colon = text.indexOf(":");
    const field = colon < 0 ? text : text.slice(0, colon);
    let value = colon < 0 ? "" : text.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value || "message";
    if (field === "data") data.push(value);
  }
  function drain(final) {
    let index;
    while ((index = buffer.search(/[\r\n]/)) !== -1) {
      if (!final && buffer[index] === "\r" && index === buffer.length - 1) break;
      const width = buffer[index] === "\r" && buffer[index + 1] === "\n" ? 2 : 1;
      const text = buffer.slice(0, index);
      buffer = buffer.slice(index + width);
      line(text);
    }
  }
  return {
    feed(chunk) { buffer += chunk; drain(false); },
    // An unterminated frame is intentionally NOT dispatched at EOF.
    end() { drain(true); buffer = ""; data = []; },
  };
}

function errorText(value) {
  if (value == null) return "Unknown error";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(errorText).join("; ");
  if (typeof value === "object") {
    return Object.entries(value).map(([key, val]) => `${key}: ${errorText(val)}`).join(" · ");
  }
  return String(value);
}

async function responseError(response) {
  const body = await response.json().catch(() => ({}));
  const detail = errorText(body.detail || body.error || response.statusText || "Request failed");
  const quota = response.status === 429 ? "Quota or rate limit reached. Check tenant usage. " : "";
  const retry = response.headers.get("retry-after");
  const error = new Error(`${quota}${detail} (HTTP ${response.status})${retry ? ` · Retry after ${retry}s` : ""}`);
  error.status = response.status;
  return error;
}

// Completion is only successful after an explicit done event, never just EOF.
async function consumeAskStream(response, onEvent, signal) {
  if (!response.ok) throw await responseError(response);
  if (!(response.headers.get("content-type") || "").includes("text/event-stream") || !response.body) {
    throw new Error("Expected an event stream from the backend.");
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let completed = false;
  const parser = createSSEParser((event, data) => {
    if (completed) return;
    if (signal?.aborted) throw new DOMException("Stopped", "AbortError");
    if (!["start", "stage", "meta", "token", "done", "error"].includes(event)) return;
    const obj = JSON.parse(data);
    if (event === "error") throw new Error(errorText(obj.error || obj.detail || obj));
    onEvent(event, obj);
    if (event === "done") completed = true;
  });
  const abort = () => { reader.cancel().catch(() => {}); };
  signal?.addEventListener("abort", abort, { once: true });
  try {
    while (!completed) {
      if (signal?.aborted) throw new DOMException("Stopped", "AbortError");
      const { done, value } = await reader.read();
      if (done) { parser.feed(decoder.decode()); parser.end(); break; }
      parser.feed(decoder.decode(value, { stream: true }));
    }
    if (signal?.aborted) throw new DOMException("Stopped", "AbortError");
    if (!completed) throw new Error("Stream interrupted before final confirmation. Retry or reopen this conversation to check saved messages.");
  } finally {
    signal?.removeEventListener("abort", abort);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

// Dependency-free Node tests can import the parser without a DOM or boot fetches.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { createSSEParser, consumeAskStream, errorText };
}
if (typeof document !== "undefined") initApp();

function initApp() {
  const $ = (selector) => document.querySelector(selector);
  const state = {
    ready: false, busy: false, navigating: false, uploading: false,
    conversationId: null, documentIds: [], versionMode: "latest", view: 0,
    boundDocumentId: null,
    requestedScope: { document_id: null, document_ids: null, version_mode: "latest" },
    documents: [], conversations: [], messages: [], controller: null,
    jobs: new Map(), watchedJobs: new Set(), docsLoad: null,
  };
  const terminalJobs = new Set(["done", "error", "cancelled"]);
  const activeDocs = new Set(["pending", "processing", "queued", "cancel_requested"]);
  const enc = encodeURIComponent;
  const node = (tag, text, className) => {
    const el = document.createElement(tag);
    if (text != null) el.textContent = String(text);
    if (className) el.className = className;
    return el;
  };
  const button = (text, action, className = "small-btn") => {
    const el = node("button", text, className);
    el.type = "button";
    el.onclick = async () => { try { await action(el); } catch (e) { toast(e.message, "err"); } };
    return el;
  };
  const navButton = (text, action, className) => {
    const el = button(text, () => { if (canNavigate()) return action(); }, className);
    el.dataset.navigation = "";
    el.disabled = !state.ready || state.busy || state.navigating;
    return el;
  };
  function toast(text, kind = "") {
    const el = node("div", text, `toast ${kind}`);
    el.setAttribute("role", kind === "err" ? "alert" : "status");
    document.body.append(el);
    setTimeout(() => el.remove(), 7000);
  }
  // Answers are withheld until citation/grounding checks pass, so the gap
  // between "asked" and "first token" can be long; a spinner here signals
  // active work instead of reading like a stalled request.
  function setStreamStatus(text, spinning = false) {
    const el = $("#streamStatus");
    el.replaceChildren();
    if (spinning) el.append(node("span", null, "spinner"));
    if (text) el.append(document.createTextNode(text));
  }
  async function api(path, options = {}) {
    const headers = { ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }), ...options.headers };
    const response = await fetch(path, { ...options, credentials: "include", headers });
    if (!response.ok) throw await responseError(response);
    return response.status === 204 ? null : response.json();
  }
  function controls() {
    const locked = !state.ready || state.busy || state.navigating;
    document.querySelectorAll("[data-navigation]").forEach(el => {
      el.disabled = locked || (el.dataset.retryView != null && el.dataset.retryView !== String(state.view));
    });
    const hasText = $("#prompt").value.trim().length > 0;
    $("#sendBtn").disabled = locked || !hasText;

    $("#prompt").disabled = !state.ready || state.navigating;
    $("#stopBtn").hidden = !state.busy;
    $("#stopBtn").disabled = !state.busy || !!state.controller?.signal.aborted;
    $("#fileInput").disabled = !state.ready || state.uploading;
    $("#dropzone").setAttribute("aria-busy", String(state.uploading));
    $("#uploadLabel").setAttribute("aria-disabled", String(!state.ready || state.uploading));
  }
  function canNavigate() {
    if (!state.ready || state.busy || state.navigating) {
      toast("Wait for the current request or stop it before changing conversations or search scope.");
      return false;
    }
    return true;
  }
  function scopeSnapshot() {
    return { ...state.requestedScope,
      document_ids: state.requestedScope.document_ids === null ? null : [...state.requestedScope.document_ids] };
  }
  function applyScope(scope) {
    // null means implicit scope; [] is an explicitly empty selection. Neither
    // should be replaced with the IDs that happened to resolve on a prior ask.
    state.requestedScope = {
      document_id: scope.document_id || null,
      document_ids: Array.isArray(scope.document_ids) ? [...scope.document_ids] : null,
      version_mode: scope.version_mode === "all" ? "all" : "latest",
    };
    if (state.boundDocumentId) {
      // The server binding wins over stale cached/legacy explicit IDs. An
      // explicit [] still means no eligible documents, even in a bound chat.
      if (state.requestedScope.document_id) state.requestedScope.document_id = state.boundDocumentId;
      if (state.requestedScope.document_ids?.length) state.requestedScope.document_ids = [state.boundDocumentId];
    }
    state.documentIds = state.requestedScope.document_ids !== null ? [...state.requestedScope.document_ids]
      : state.requestedScope.document_id ? [state.requestedScope.document_id]
      : state.boundDocumentId ? [state.boundDocumentId] : [];
    state.versionMode = state.requestedScope.version_mode;
  }
  function changeScope(ids, versionMode = state.versionMode) {
    const before = scopeLabel();
    if (state.boundDocumentId && !(ids?.length === 1 && ids[0] === state.boundDocumentId)) {
      resetChat();
      toast("Started a new chat: the previous conversation is bound to its original document.");
    }
    applyScope({ document_id: ids?.length === 1 ? ids[0] : null,
      document_ids: ids, version_mode: versionMode });
    updateScope(); renderDocs();
    noteScopeChange(before);
  }
  function saveScope(cid, scope) {
    if (!cid) return;
    try { sessionStorage.setItem(`vrag_scope_${cid}`, JSON.stringify(scope)); } catch (_) { /* Storage may be disabled. */ }
  }
  function restoreScope(conversation, messages) {
    let saved;
    try { saved = JSON.parse(sessionStorage.getItem(`vrag_scope_${conversation.id}`) || "null"); } catch (_) {}
    const meta = [...messages].reverse().find(m => m.role === "assistant" && (m.meta?.requested_scope || m.meta?.scope))?.meta;
    state.boundDocumentId = conversation.document_id || null;
    applyScope(meta?.requested_scope || saved || meta?.scope || conversation);
    updateScope();
  }
  function updateScope() {
    $("#versionMode").value = state.versionMode;
    const chip = $("#scopeChip");
    chip.hidden = false;
    chip.textContent = state.documentIds.length ? "Exact scope: " + state.documentIds.map(id => {
      const doc = state.documents.find(d => d.id === id);
      return doc ? `${doc.filename} v${doc.version}${doc.status !== "ready" ? ` (${doc.status})` : ""}` : "Unavailable document";
    }).join(", ") : state.requestedScope.document_ids !== null ? "Empty exact scope: no documents selected"
      : state.versionMode === "all" ? "All ready versions" : "Latest ready version per filename";
    if (state.boundDocumentId) chip.textContent += " · Conversation bound to its original document";
    saveScope(state.conversationId, scopeSnapshot());
  }
  function scopeLabel() {
    if (state.documentIds.length) return "Scope: " + state.documentIds.map(id => {
      const doc = state.documents.find(d => d.id === id);
      return doc ? `${doc.filename} v${doc.version}` : "unavailable document";
    }).join(", ");
    if (state.requestedScope.document_ids !== null) return "Scope: no documents selected";
    return state.versionMode === "all" ? "Scope: all ready versions" : "Scope: latest ready version per filename";
  }
  // Only worth recording once a conversation actually has messages in it —
  // narrowing scope before the first question is just normal setup, not a
  // change the transcript needs to explain.
  function noteScopeChange(previousLabel) {
    if (!state.messages.length) return;
    const label = scopeLabel();
    if (label === previousLabel) return;
    const message = { role: "system", content: `${label} — earlier answers above were grounded under the previous scope.` };
    state.messages.push(message); renderMessage(message); scrollChat();
  }

  function renderDocs() {
    const list = $("#docList"); list.replaceChildren();
    if (!state.documents.length) list.append(node("li", "No documents yet", "doc-item empty-item"));
    for (const doc of state.documents) {
      const li = node("li", null, "doc-item");
      const label = node("label", null, "doc-choice");
      const input = node("input"); input.type = "checkbox"; input.dataset.navigation = "";
      input.checked = state.documentIds.includes(doc.id);
      input.setAttribute("aria-label", `Search ${doc.filename} version ${doc.version}`);
      input.onchange = () => {
        if (!canNavigate()) { input.checked = state.documentIds.includes(doc.id); return; }
        const ids = input.checked ? [...state.documentIds, doc.id] : state.documentIds.filter(id => id !== doc.id);
        changeScope(ids.length ? ids : null);
      };
      const titleSpan = node("span", `${doc.filename} · v${doc.version}`, "doc-title");
      titleSpan.title = doc.filename;
      label.append(input, titleSpan);
      const details = button("Details", () => openLifecycle(doc.id));
      details.setAttribute("aria-label", `Lifecycle details for ${doc.filename} version ${doc.version}`);
      const summary = node("div", null, "doc-summary");
      const statusClass = ["ready", "error", "processing", "cancelled"].includes(doc.status) ? doc.status : "processing";
      summary.append(node("span", doc.status, `doc-status ${statusClass}`), document.createTextNode(` · ${doc.pages ?? "?"} pages · ${doc.chunks ?? "?"} chunks · ${doc.attempt_count ?? 0} attempts`));
      if (doc.last_error) summary.append(node("div", errorText(doc.last_error), "error-text"));
      li.append(label, details, summary); list.append(li);
    }
    controls();
  }
  async function loadDocs(resume = true) {
    if (state.docsLoad) return state.docsLoad;
    state.docsLoad = (async () => {
      state.documents = await api("/api/documents");
      renderDocs(); updateScope();
      if (resume) for (const doc of state.documents) {
        if (doc.job_id && activeDocs.has(doc.status)) watchJob(doc.job_id, doc.id);
      }
    })();
    try { await state.docsLoad; } finally { state.docsLoad = null; }
  }
  function renderConvs() {
    const list = $("#convList"); list.replaceChildren();
    if (!state.conversations.length) list.append(node("li", "No conversations yet", "conv-item empty-item"));
    for (const conv of state.conversations) {
      const li = node("li", null, "conv-item" + (conv.id === state.conversationId ? " active" : ""));
      const title = conv.title || "Untitled conversation";
      const open = navButton(title, () => openConversation(conv.id), "conv-title");
      open.title = title;
      const remove = navButton("✕", async () => {
        if (!confirm(`Delete ${title}?`)) return;
        state.navigating = true; controls();
        try {
          await api(`/api/conversations/${enc(conv.id)}`, { method: "DELETE" });
          if (state.conversationId === conv.id) resetChat();
          await loadConvs();
        } finally { state.navigating = false; controls(); }
      }, "del");
      remove.setAttribute("aria-label", `Delete ${title}`);
      li.append(open, remove); list.append(li);
    }
  }
  async function loadConvs() {
    state.conversations = await api("/api/conversations");
    const current = state.conversations.find(c => c.id === state.conversationId);
    if (current) {
      state.boundDocumentId = current.document_id || null;
      applyScope(scopeSnapshot()); updateScope(); renderDocs();
    }
    renderConvs();
  }
  function resetChat() {
    state.view++; state.conversationId = null; state.messages = []; state.boundDocumentId = null;
    applyScope({ document_id: null, document_ids: null, version_mode: "latest" });
    $("#chat").replaceChildren(); // Deliberately empty: no welcome block.
    setStreamStatus(""); $("#crumb").textContent = "New chat"; $("#jumpLatest").hidden = true;
    updateScope(); renderDocs(); renderConvs();
  }
  async function openConversation(cid) {
    if (!canNavigate()) return;
    state.navigating = true; controls();
    try {
      const messages = await api(`/api/conversations/${enc(cid)}/messages`);
      state.view++; state.conversationId = cid; state.messages = messages;
      restoreScope(state.conversations.find(c => c.id === cid) || { id: cid }, messages);
      $("#chat").replaceChildren(); messages.forEach(renderMessage);
      $("#crumb").textContent = "Conversation"; setStreamStatus("");
      renderDocs(); renderConvs(); scrollChat(true);
    } finally { state.navigating = false; controls(); }
  }

  function evidenceButton(source, text) {
    const el = button(text, () => openEvidence(source), "citation");
    el.disabled = !source.document_id || !source.evidence_id;
    el.setAttribute("aria-label", `${source.label || "Evidence"}: ${source.filename || "document"}, page ${source.page ?? "unknown"}`);
    return el;
  }
 function renderAnswerText(container, text, sources) {
    container.replaceChildren();
    const known = new Map((sources || []).map(s => [s.label, s]));
    const pattern = /\[(E\d+)\]/g;
    let previous = 0;

    // Helper to safely parse **bold** text into HTML strong tags
    function appendFormatted(str) {
      const parts = str.split(/(\*\*.*?\*\*)/g);
      for (const part of parts) {
        if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
          const strong = document.createElement('strong');
          strong.textContent = part.slice(2, -2);
          container.append(strong);
        } else if (part) {
          container.append(document.createTextNode(part));
        }
      }
    }

    for (const match of String(text).matchAll(pattern)) {
      appendFormatted(text.slice(previous, match.index));
      const source = known.get(match[1]);
      container.append(source ? evidenceButton(source, match[0]) : document.createTextNode(match[0]));
      previous = match.index + match[0].length;
    }
    appendFormatted(String(text).slice(previous));
  }
  function renderMessage(message) {
    if (message.role === "system") {
      const el = node("article", null, "msg system");
      el.append(node("div", message.content, "msg-content"));
      $("#chat").append(el);
      return el;
    }
    const el = node("article", null, `msg ${message.role === "user" ? "user" : "assistant"}${message.error ? " error" : ""}`);
    const body = node("div", null, "msg-body"), content = node("div", null, "msg-content");
    el.append(node("div", message.role === "user" ? "U" : "A", "msg-avatar"), body); body.append(content);
    if (message.role === "user") content.textContent = message.content;
    else {
      const meta = message.meta || {};
      renderAnswerText(content, message.content || "", meta.sources);
      const metadata = node("div", null, "msg-meta");
      const abstained = meta.status === "abstained" || meta.abstained || meta.answerable === false;
      if (abstained) body.append(node("p", `Abstained: ${meta.abstention_reason || "Insufficient supporting evidence."}`, "abstention"));
      else if (meta.status === "answered") metadata.append(node("span", "Answered · verified", "chip"));
      if (meta.confidence_available === true && typeof meta.confidence === "number") {
        const band = String(meta.confidence_band || "").toUpperCase();
        metadata.append(node("span", `Confidence ${Math.round(meta.confidence * 100)}% · ${band}`, `chip ${band === "LOW" ? "warn" : ""}`));
      } else if (meta.confidence_available === false || message.id) {
        metadata.append(node("span", `Confidence unavailable: ${meta.confidence_unavailable_reason || "No calibrated confidence provided."}`, "chip subtle"));
      }

      if(meta.tier){
        const tierName=meta.tier.replace("MODEL_", "").toLowerCase();
        metadata.append(node("span", `Model: ${tierName}`, "chip subtle"));
      }

      if (meta.prompt_tokens != null || meta.completion_tokens != null) {
        metadata.append(node("span", `Tokens: ${meta.prompt_tokens || 0} in · ${meta.completion_tokens || 0} out`, "chip subtle"));
      }
      
      for (const source of meta.sources || []) {
        metadata.append(evidenceButton(source, `[${source.label || "Evidence"}] ${source.filename || "Document"} · p.${source.page ?? "?"}${source.cited ? " · cited" : " · retrieved"}`));
      }
      body.append(metadata);
      if (meta.citation_warnings?.length) body.append(node("p", `Citation warnings: ${errorText(meta.citation_warnings)}`, "abstention"));
      if (meta.retrieval_query || meta.calibration != null) {
        const details = node("details", null, "answer-details"); details.append(node("summary", "Retrieval and calibration"));
        if (meta.retrieval_query) details.append(node("p", `Retrieval query: ${meta.retrieval_query}`));
        if (meta.calibration != null) details.append(node("p", `Calibration: ${errorText(meta.calibration)}`));
        body.append(details);
      }
      // ---> NEW: Wrap the entire actions creation in this IF statement <---
      if (message.id || message.error) {
        const actions = node("div", null, "msg-actions");

        // Helper function to create clean SVG icon buttons
        function createIconButton(svgInner, title, action) {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "icon-btn"; // Uses your existing clean, borderless button style
          btn.title = title;
          btn.setAttribute("aria-label", title);
          btn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${svgInner}</svg>`;
          btn.onclick = async () => {
            try { await action(btn); } catch (e) { toast(e.message, "err"); }
          };
          return btn;
        }

        // 1. Copy Button (Clipboard icon)
        const copyIcon = `<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>`;
        actions.append(createIconButton(copyIcon, "Copy text", async () => { 
          await navigator.clipboard.writeText(message.content || ""); 
          toast("Copied"); 
        }));

        // 2. Feedback Buttons (Thumb Up / Thumb Down icons)
        if (message.id) {
          const thumbUp = `<path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>`;
          const thumbDown = `<path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>`;

          const feedbackOptions = [
            [1, thumbUp, "Helpful"], 
            [-1, thumbDown, "Not helpful"]
          ];

          for (const [vote, svgPath, label] of feedbackOptions) {
            actions.append(createIconButton(svgPath, label, async b => {
              await api("/api/feedback", { method: "POST", body: JSON.stringify({ message_id: message.id, vote }) });
              b.setAttribute("aria-pressed", "true");
              b.style.color = "var(--accent)"; // Highlights the icon when clicked
              toast("Feedback saved");
            }));
          }
        }

        if (message.error && message.retry) {
          const retryButton = navButton("Retry question", () => {
          const captured = message.retry;
          if (captured.view !== state.view || captured.conversation_id !== state.conversationId) {
            toast("Reopen the original conversation before retrying.", "err"); return;
          }
          return ask(captured.question, captured);
          }, "retry-btn");
          retryButton.dataset.retryView = String(message.retry.view);
          actions.append(retryButton);
        }
        
        body.append(actions);
      }
    }
    $("#chat").append(el); return el;
  }
  function replaceMessage(el, message) { const replacement = renderMessage(message); el.replaceWith(replacement); return replacement; }
  // A streaming answer used to force-scroll on every token, which yanked the
  // view back down even if the user had deliberately scrolled up to reread
  // something earlier. Now it only follows the bottom if the user was
  // already there; otherwise "New messages" appears instead of hijacking
  // their scroll position.
  function nearBottom() {
    const el = $("#chat");
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }
  function updateJumpButton() { $("#jumpLatest").hidden = nearBottom(); }
  function scrollChat(force = false) {
    const el = $("#chat");
    if (force || nearBottom()) el.scrollTop = el.scrollHeight;
    updateJumpButton();
  }
  $("#chat").addEventListener("scroll", updateJumpButton);
  $("#jumpLatest").onclick = () => scrollChat(true);

  // Modal requests carry a generation token; closing/replacing a dialog makes
  // late network responses harmless. Focus is trapped and restored on close.
  let modalGeneration = 0, modalFocus = null;
  function showModal(title) {
    if ($("#modal").hidden) modalFocus = document.activeElement;
    const generation = ++modalGeneration;
    $("#modalTitle").textContent = title; $("#modalBody").replaceChildren(); $("#modal").hidden = false;
    $(".sidebar").inert = true; $(".main").inert = true;
    $("#modalClose").focus();
    return { body: $("#modalBody"), current: () => generation === modalGeneration && !$("#modal").hidden };
  }
  function closeModal() {
    modalGeneration++; $("#modal").hidden = true; $(".sidebar").inert = false; $(".main").inert = false;
    if (modalFocus?.isConnected && !modalFocus.disabled) modalFocus.focus(); else $("#newChat").focus();
  }
  $("#modalClose").onclick = closeModal;
  $("#modal").onclick = event => { if (event.target === $("#modal")) closeModal(); };
  $("#modal").addEventListener("keydown", event => {
    if (event.key === "Escape") { event.preventDefault(); closeModal(); }
    if (event.key !== "Tab") return;
    const items = [...$("#modal").querySelectorAll("button:not(:disabled), select:not(:disabled), input:not(:disabled), [tabindex='0'], summary")].filter(el => !el.hidden && el.getClientRects().length);
    const first = items[0], last = items[items.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
  async function openEvidence(source) {
    const modal = showModal("Evidence"); modal.body.append(node("p", "Loading evidence…"));
    try {
      const evidence = await api(`/api/documents/${enc(source.document_id)}/evidence/${enc(source.evidence_id)}`);
      if (!modal.current()) return;
      modal.body.replaceChildren(node("h3", `${evidence.filename || source.filename} · page ${evidence.page ?? source.page}`));
      modal.body.append(node("blockquote", evidence.quote || evidence.preview || source.quote || source.preview || "No quote available."));
      const precise = Array.isArray(evidence.bbox) && evidence.bbox.length === 4 && evidence.bbox.every(Number.isFinite);
      modal.body.append(node("p", precise ? "Region-level evidence: the page image highlights the recorded bounding box, not necessarily every quoted word." : "Page-level evidence only: no accurate bounding box is available. Use the quote to locate the supporting text."));
      if (evidence.page_image_url) {
        const url = new URL(evidence.page_image_url, location.origin);
        const prefix = `/api/documents/${enc(source.document_id)}/`;
        if (url.origin !== location.origin || !url.pathname.startsWith(prefix)) throw new Error("The backend returned an unsupported evidence image URL.");
        const image = node("img"); image.alt = `${evidence.filename || source.filename}, page ${evidence.page ?? source.page}${precise ? ", highlighted evidence region" : ", page-level evidence"}`;
        image.onerror = () => { if (modal.current()) { image.remove(); modal.body.append(node("p", "Page image unavailable. The evidence quote remains available.", "error-text")); } };
        image.src = url.href; modal.body.append(image);
      } else modal.body.append(node("p", "No page image is available; refer to the quote."));
    } catch (e) { if (modal.current()) modal.body.append(node("p", e.message, "error-text")); }
  }
  const dateText = value => value == null ? "—" : new Date(typeof value === "number" ? value * 1000 : value).toLocaleString();
  const bytes = value => value == null ? "—" : `${(Number(value) / (1024 * 1024)).toFixed(1)} MB`;
  async function openLifecycle(id) {
    const modal = showModal("Document lifecycle"); modal.body.append(node("p", "Loading document…"));
    try {
      const { document: doc, jobs } = await api(`/api/documents/${enc(id)}/detail`);
      if (!modal.current()) return;
      modal.body.replaceChildren(node("h3", `${doc.filename} · v${doc.version}`));
      modal.body.append(node("p", `${doc.status} · ${bytes(doc.size_bytes)} · ${doc.pages ?? "?"} pages · ${doc.chunks ?? "?"} chunks · ${doc.attempt_count ?? 0} attempts`), node("p", `Created: ${dateText(doc.created_at)} · Updated: ${dateText(doc.updated_at)}`));
      if (doc.last_error) modal.body.append(node("p", errorText(doc.last_error), "error-text"));
      const actions = node("div", null, "dialog-actions");
      actions.append(button("Refresh details", () => openLifecycle(id)));
      for (const action of ["retry", "cancel"]) {
        const allowed = action === "retry" ? ["error", "cancelled"].includes(doc.status) : activeDocs.has(doc.status);
        const control = button(action === "retry" ? "Retry ingestion" : "Cancel ingestion", async b => {
          b.disabled = true;
          try {
            const result = await api(`/api/documents/${enc(id)}/${action}`, { method: "POST" });
            if (result.job_id) watchJob(result.job_id, id, true);
            toast(result.cancel_requested ? "Cancellation requested; waiting for the worker to stop." : "Retry queued.");
            await Promise.allSettled([loadDocs(), loadUsage()]);
            if (modal.current()) await openLifecycle(id);
          } catch (e) { b.disabled = !allowed; throw e; }
        });
        control.disabled = !allowed; actions.append(control);
      }
      actions.append(navButton("Delete document", async () => {
        if (!confirm(`Delete ${doc.filename} version ${doc.version}?`)) return;
        state.navigating = true; controls();
        try {
          await api(`/api/documents/${enc(id)}`, { method: "DELETE" });
          // Keep pinned IDs (including deleted/unavailable versions) rather
          // than silently broadening the next ask to all documents.
          if (modal.current()) closeModal();
          await Promise.allSettled([loadDocs(), loadUsage()]);
        } finally { state.navigating = false; controls(); }
      }));
      modal.body.append(actions);
      if (!jobs?.length) modal.body.append(node("p", "No recorded ingestion attempts."));
      for (const job of jobs || []) {
        const card = node("section", null, "job-card"), result = job.result || {};
        card.append(node("h4", `Job ${job.id} · ${job.status}`), node("p", `${job.stage || "No stage reported"} · ${job.progress ?? 0}/${job.total ?? "?"} · ${dateText(job.updated_at)}`));
        if (job.message) card.append(node("p", errorText(job.message), job.status === "error" ? "error-text" : ""));
        if (result.failed_page) card.append(node("p", `Failed/interrupted page: ${result.failed_page}`, "error-text"));
        if (result.pages_without_content?.length) card.append(node("p", `Pages without usable content: ${result.pages_without_content.join(", ")}`, "abstention"));
        if (result.warnings?.length) card.append(node("p", errorText(result.warnings), "abstention"));
        if (result.page_statuses) card.append(node("pre", `Per-page status: ${errorText(result.page_statuses)}`));
        card.append(node("p", "Only worker-reported page status is shown; progress is not proof that every page was indexed.", "hint"));
        modal.body.append(card);
      }
    } catch (e) { if (modal.current()) modal.body.append(node("p", e.message, "error-text")); }
  }

  function openCompare() {
    const modal = showModal("Compare document versions");
    modal.body.append(node("p", "Extracted-text comparison only, not a visual pixel diff. Layout, images, OCR omissions and extraction changes may not be reflected accurately.", "abstention"));
    const form = node("div", null, "compare-controls"), selects = [];
    for (const side of ["Left", "Right"]) {
      const label = node("label", `${side} document version`), select = node("select");
      select.id = `compare${side}`;
      for (const doc of state.documents.filter(d => d.status === "ready")) {
        const option = node("option", `${doc.filename} · v${doc.version}`); option.value = doc.id; select.append(option);
      }
      label.append(select); form.append(label); selects.push(select);
    }
    const chosen = state.documentIds.filter(id => state.documents.some(d => d.id === id && d.status === "ready"));
    if (chosen[0]) selects[0].value = chosen[0];
    if (chosen[1]) selects[1].value = chosen[1]; else if (selects[1].options.length > 1) selects[1].selectedIndex = 1;
    const output = node("div");
    const compare = button("Compare", () => runComparison(0)); form.append(compare); modal.body.append(form, output);
    let revision = 0;
    selects.forEach(select => { select.onchange = () => { revision++; output.replaceChildren(); }; });
    async function runComparison(offset) {
      const [left, right] = selects.map(select => select.value);
      if (!left || !right || left === right) { toast("Choose two distinct ready document versions.", "err"); return; }
      const request = ++revision; compare.disabled = true;
      output.replaceChildren(node("p", "Comparing extracted pages…"));
      try {
        const query = new URLSearchParams({ left_id: left, right_id: right, offset: String(offset), limit: "20" });
        const result = await api(`/api/documents/compare?${query}`);
        if (!modal.current() || request !== revision) return;
        const leftTitle = `${result.left.filename} v${result.left.version}`, rightTitle = `${result.right.filename} v${result.right.version}`;
        output.replaceChildren(node("h3", `${leftTitle} ↔ ${rightTitle}`), node("p", `${result.total_pages} extracted pages · showing ${result.pages.length ? offset + 1 : 0}–${offset + result.pages.length}`));
        if (result.truncated) output.append(node("p", "This is a partial view of the comparison.", "abstention"));
        if (result.text_truncated) output.append(node("p", "Text previews are truncated. Changed flags compare full extracted text, including omitted portions.", "abstention"));
        for (const page of result.pages) {
          const section = node("section", null, "compare-page");
          section.append(node("h4", `Page ${page.page} · ${page.changed ? "Changed" : "Unchanged"}`, page.changed ? "changed" : ""));
          const columns = node("div", null, "compare-columns");
          for (const [title, text] of [[leftTitle, page.left_text], [rightTitle, page.right_text]]) {
            const col = node("div"); col.append(node("h5", title), node("pre", text || "No extracted text in this preview.")); columns.append(col);
          }
          section.append(columns); output.append(section);
        }
        const paging = node("div", null, "dialog-actions");
        if (offset > 0) paging.append(button("Previous pages", () => runComparison(Math.max(0, offset - 20))));
        if (result.next_offset != null) paging.append(button("Next pages", () => runComparison(result.next_offset)));
        output.append(paging);
      } catch (e) { if (modal.current() && request === revision) output.replaceChildren(node("p", e.message, "error-text")); }
      finally { if (modal.current()) compare.disabled = false; }
    }
  }

  async function loadUsage() {
    try {
      const snapshot = await api("/api/usage"), container = $("#usage");
      container.replaceChildren(node("p", `${snapshot.day} · ${snapshot.timezone}`));
      for (const [key, label] of [["documents", "Documents"], ["storage_bytes", "Storage"], ["daily_asks", "Daily asks"], ["daily_pages", "Daily pages"],["input_tokens", "Input tokens"],["output_tokens", "Output tokens"]]) {
        const format = key === "storage_bytes" ? bytes : value => value == null ? "—" : String(value);
        const limit = snapshot.limits?.[key];
        const remaining = snapshot.remaining?.[key];
        container.append(node("div", `${label}: ${format(snapshot.usage?.[key])} / ${limit == null ? "not specified" : format(limit)} · ${remaining == null ? "remaining not specified" : `${format(remaining)} remaining`}`, "usage-row"));
      }
    } catch (e) { $("#usage").replaceChildren(node("p", `Usage unavailable: ${e.message}`, "error-text")); }
  }
  async function pingHealth() {
    try {
      const result = await api("/readyz");
      $("#statusDot").className = `status-dot ${result.ready ? "ok" : "err"}`;
      $("#statusText").textContent = result.ready ? "ready" : `degraded${result.details ? ` · ${errorText(result.details)}` : ""}`;
    } catch (e) { $("#statusDot").className = "status-dot err"; $("#statusText").textContent = e.message; }
  }

  function jobProgress(job, docId) {
    const doc = state.documents.find(d => d.id === docId);
    $("#dzProgress").hidden = false;
    $("#dzText").textContent = `${doc?.filename || "Document"}: ${job.status} · ${job.stage || ""} · ${job.progress ?? 0}/${job.total ?? "?"}${job.message ? ` · ${errorText(job.message)}` : ""}`;
    if (job.total > 0) $("#dzBar").value = Math.min(100, Math.max(0, 100 * (job.progress || 0) / job.total));
    else $("#dzBar").removeAttribute("value");
  }
  function watchJob(jobId, docId, force = false) {
    if (!jobId || state.jobs.has(jobId) || (!force && state.watchedJobs.has(jobId))) return;
    state.watchedJobs.add(jobId);
    const watcher = { source: null, timer: null, controller: null, stopped: false, polling: false, attempts: 0 };
    state.jobs.set(jobId, watcher);
    function stop() {
      watcher.stopped = true; watcher.source?.close(); clearTimeout(watcher.timer); watcher.controller?.abort(); state.jobs.delete(jobId);
    }
    function update(job) {
      if (watcher.stopped) return;
      jobProgress(job, docId);
      if (terminalJobs.has(job.status)) {
        stop(); toast(`Ingestion ${job.status}${job.message ? `: ${errorText(job.message)}` : ""}`, job.status === "error" ? "err" : "");
        Promise.allSettled([loadDocs(false), loadUsage()]);
      }
    }
    async function poll() {
      if (watcher.stopped) return;
      if (watcher.attempts++ >= 40) {
        stop(); $("#dzText").textContent = "Progress monitoring paused. The worker may still be running; Refresh documents to resume."; return;
      }
      watcher.controller = new AbortController();
      const timeout = setTimeout(() => watcher.controller?.abort(), 8000);
      try { update(await api(`/api/jobs/${enc(jobId)}`, { signal: watcher.controller.signal })); }
      catch (e) { if (!watcher.stopped) $("#dzText").textContent = `Progress unavailable: ${e.message}. Retrying status…`; }
      finally { clearTimeout(timeout); }
      if (!watcher.stopped) watcher.timer = setTimeout(poll, 3000);
    }
    function fallback() {
      if (watcher.stopped || watcher.polling) return;
      watcher.polling = true; watcher.source?.close(); clearTimeout(watcher.timer); poll();
    }
    try {
      watcher.source = new EventSource(`/api/jobs/${enc(jobId)}/stream`, { withCredentials: true });
      for (const type of ["progress", "end", "done", "cancelled"]) watcher.source.addEventListener(type, event => {
        try { update(JSON.parse(event.data)); if (type !== "progress" && !watcher.stopped) fallback(); } catch (_) { fallback(); }
      });
      watcher.source.onerror = fallback;
      // Bound even a silently stalled SSE connection, then poll for at most 40 attempts.
      watcher.timer = setTimeout(fallback, 60000);
    } catch (_) { fallback(); }
  }
  async function uploadFile(file) {
    if (!state.ready || state.uploading) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) { toast("Only PDF files are accepted.", "err"); return; }
    if (file.size > 50 * 1024 * 1024) { toast("Maximum upload size is 50 MB.", "err"); return; }
    state.uploading = true; controls(); $("#dzProgress").hidden = false; $("#dzBar").removeAttribute("value");
    $("#dzText").textContent = `Uploading ${file.name}…`;
    try {
      const form = new FormData(); form.append("file", file);
      const result = await api("/api/documents/upload", { method: "POST", body: form });
      $("#dzText").textContent = "Upload accepted; processing queued.";
      watchJob(result.job_id, result.document_id, true);
      await Promise.allSettled([loadDocs(), loadUsage()]);
    } catch (e) {
      $("#dzText").textContent = `Upload failed: ${e.message}`; toast(e.message, "err");
      // A rejected upload may already have registered an errored document.
      await Promise.allSettled([loadDocs(), loadUsage()]);
    }
    finally { state.uploading = false; controls(); }
  }

  async function ask(question, retry = null) {
    // Guard both click and Enter; retries also retain their original scope/view.
    if (!state.ready || state.busy || state.navigating || !question.trim()) return;
    const captured = retry ? { ...retry, document_ids: Array.isArray(retry.document_ids) ? [...retry.document_ids] : null } : { question, conversation_id: state.conversationId, ...scopeSnapshot(), view: state.view };
    if (captured.view !== state.view || captured.conversation_id !== state.conversationId) return;
    state.busy = true; state.controller = new AbortController(); controls();
    const signal = state.controller.signal;
    applyScope(captured); updateScope(); renderDocs();
    if (!retry) { const user = { role: "user", content: question }; state.messages.push(user); renderMessage(user); scrollChat(true); }
    const assistant = { role: "assistant", content: "", meta: {} }; state.messages.push(assistant);
    let element = renderMessage(assistant);
    setStreamStatus("Retrieving evidence. Output is shown only after verification, not as live provider deltas.", true);
    const payload = { question: captured.question, conversation_id: captured.conversation_id, document_id: captured.document_id, document_ids: captured.document_ids, version_mode: captured.version_mode };
    try {
      const response = await fetch("/api/ask/stream", { method: "POST", credentials: "include", signal,
        headers: { "Content-Type": "application/json", "Accept": "text/event-stream" }, body: JSON.stringify(payload) });
      await consumeAskStream(response, (event, obj) => {
        if (captured.view !== state.view || signal.aborted) return;
        if (event === "start" && obj.conversation_id) {
          if (!captured.conversation_id) state.boundDocumentId = captured.document_id
            || (captured.document_ids?.length === 1 ? captured.document_ids[0] : null);
          state.conversationId = obj.conversation_id; captured.conversation_id = obj.conversation_id;
          applyScope(captured); updateScope();
        } else if (event === "stage") {
          setStreamStatus(obj.stage === "verifying" ? "Verifying answer and citations before showing finalized output…" : `Preparing verified answer · ${obj.stage}`, true);
        } else if (event === "meta") {
          assistant.meta = { ...assistant.meta, ...obj };
        } else if (event === "token") {
          const stick = nearBottom();
          assistant.content += obj.text || "";
          element.querySelector(".msg-content").textContent = assistant.content;
          setStreamStatus("Receiving verified, finalized answer chunks…", true);
          scrollChat(stick);
        } else if (event === "done") {
          if (typeof obj.answer !== "string") throw new Error("Final response is missing the confirmed answer.");
          const stick = nearBottom();
          assistant.content = obj.answer; assistant.id = obj.message_id;
          const requested = { document_id: captured.document_id, document_ids: captured.document_ids, version_mode: captured.version_mode };
          assistant.meta = { ...assistant.meta, ...obj,
            requested_scope: obj.requested_scope || assistant.meta.requested_scope || requested, scope: requested };
          if (obj.conversation_id) state.conversationId = captured.conversation_id = obj.conversation_id;
          applyScope(assistant.meta.requested_scope); updateScope();
          element = replaceMessage(element, assistant);
          setStreamStatus(obj.abstained || obj.status === "abstained" || obj.answerable === false ? "Complete · abstained because evidence was insufficient." : "Complete · verified answer received.", false);
          $("#crumb").textContent = "Conversation"; scrollChat(stick);
        }
      }, signal);
    } catch (e) {
      if (captured.view === state.view) {
        const stick = nearBottom();
        assistant.content = signal.aborted ? "Stopped. The server may still finish processing; reopen this conversation to check saved messages before retrying." : `Request failed: ${e.message}`;
        assistant.error = true; assistant.meta = {}; assistant.retry = { ...captured };
        element = replaceMessage(element, assistant);
        setStreamStatus(signal.aborted ? "Stopped · completion not confirmed." : "Failed · completion not confirmed.", false);
        scrollChat(stick);
      }
    } finally {
      state.busy = false; state.controller = null; controls();
      // Always reload: start already assigned the conversation before done.
      await Promise.allSettled([loadConvs(), loadUsage()]);
    }
  }
  function send() {
    if (!state.ready || state.busy || state.navigating) return;
    const question = $("#prompt").value.trim(); if (!question) return;
    $("#prompt").value = ""; $("#prompt").style.height = "auto"; ask(question);
  }
  $("#sendBtn").onclick = send;
  $("#prompt").addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
  });
  $("#prompt").addEventListener("input", () => { $("#prompt").style.height = "auto"; $("#prompt").style.height = Math.min(200, $("#prompt").scrollHeight) + "px"; controls(); });
  $("#stopBtn").onclick = () => {
    state.controller?.abort(); controls();
    if (state.busy) setStreamStatus("Stopping · completion not confirmed.", true);
  };
  $("#newChat").onclick = () => { if (canNavigate()) resetChat(); };
  $("#clearScope").onclick = () => { if (canNavigate()) changeScope(null); };
  $("#versionMode").onchange = event => {
    if (canNavigate()) {
      const before = scopeLabel();
      applyScope({ ...scopeSnapshot(), version_mode: event.target.value }); updateScope();
      noteScopeChange(before);
    } else event.target.value = state.versionMode;
  };
  $("#compareBtn").onclick = openCompare;
  $("#refreshDocs").onclick = async () => {
    if (!state.ready) return;
    $("#refreshDocs").disabled = true;
    try { state.watchedJobs.clear(); await loadDocs(); await loadUsage(); }
    catch (e) { toast(e.message, "err"); }
    finally { $("#refreshDocs").disabled = false; }
  };
  $("#fileInput").onchange = event => { const file = event.target.files?.[0]; event.target.value = ""; if (file) uploadFile(file); };
  for (const type of ["dragenter", "dragover", "dragleave", "drop"]) $("#dropzone").addEventListener(type, event => {
    event.preventDefault(); $("#dropzone").classList.toggle("dragover", type === "dragenter" || type === "dragover");
    if (type === "drop" && event.dataTransfer.files?.[0]) uploadFile(event.dataTransfer.files[0]);
  });
  $("#dropzone").addEventListener("click", event => {
    if (event.target.closest("label")) return; // the "Browse files" label already opens the picker natively
    if (!state.ready || state.uploading) return;
    $("#fileInput").click();
  });
  $("#dropzone").addEventListener("keydown", event => {
    if (event.key !== "Enter" && event.key !== " ") return;
    if (event.target.closest("label")) return;
    event.preventDefault(); // a div with role="button" gets no native Enter/Space activation
    if (!state.ready || state.uploading) return;
    $("#fileInput").click();
  });
  $("#exportBtn").onclick = () => {
    if (!state.messages.length) { toast("Nothing to export."); return; }
    const text = state.messages.map(m => `## ${m.role}\n\n${m.content}\n\n${(m.meta?.sources || []).map(s => `[${s.label}] ${s.filename} · p.${s.page} · ${s.quote || s.preview || ""}`).join("\n")}`).join("\n\n");
    const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
    const anchor = node("a"); anchor.href = url; anchor.download = `vrag-conversation-${Date.now()}.md`; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  try { document.documentElement.dataset.theme = localStorage.getItem("vrag_theme") === "light" ? "light" : "dark"; } catch (_) {}
  function syncThemeToggle() {
    $("#themeToggle").setAttribute("aria-pressed", String(document.documentElement.dataset.theme === "light"));
  }
  syncThemeToggle();
  $("#themeToggle").onclick = () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("vrag_theme", theme); } catch (_) {}
    syncThemeToggle();
  };
  async function boot() {
    controls(); $("#chat").replaceChildren();
    try {
      // Establish the session cookie BEFORE any concurrent boot requests.
      await api("/api/session");
      state.ready = true; controls();
      const results = await Promise.allSettled([pingHealth(), loadDocs(), loadConvs(), loadUsage()]);
      const failed = results.filter(r => r.status === "rejected");
      if (failed.length) {
        $("#bootStatus").hidden = false;
        $("#bootStatus").textContent = failed.map(r => r.reason.message).join(" · ");
      }
      setInterval(pingHealth, 30000);
    } catch (e) {
      $("#bootStatus").hidden = false;
      $("#bootStatus").textContent = `Session setup failed: ${e.message}. Reload to retry.`;
      controls();
    }
  }
  boot();
}