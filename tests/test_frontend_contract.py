"""Offline frontend contracts; stdlib-only, with optional Node parser execution.

No application/API imports, external services, or new test dependencies. Node
checks use synthetic streams and can also be run independently of Python.
"""
from __future__ import annotations

import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web" / "app.js"

# Kept in this owned test file rather than introducing another JS test file.
NODE_TESTS = r'''
const assert = require("node:assert/strict");
const { createSSEParser, consumeAskStream, errorText } = require(process.argv[1]);

function parse(chunks) {
  const events = [];
  const parser = createSSEParser((type, data) => events.push([type, data]));
  chunks.forEach(chunk => parser.feed(chunk));
  parser.end();
  return events;
}

for (const ending of ["\n", "\r\n", "\r"]) {
  const stream = [": heartbeat", "id: 5", "event: token", "data: {\"text\":", "data: \"héllo\"}", "", "event: done", "data: {\"answer\":\"héllo\"}", "", ""].join(ending);
  const expected = [["token", '{"text":\n"héllo"}'], ["done", '{"answer":"héllo"}']];
  assert.deepEqual(parse([stream]), expected);
  // Every possible two-chunk boundary, including between CR and LF.
  for (let split = 0; split <= stream.length; split++) {
    assert.deepEqual(parse([stream.slice(0, split), stream.slice(split)]), expected);
  }
  assert.deepEqual(parse([...stream]), expected);
}
assert.deepEqual(parse(["event: token\r", "\ndata:  spaced  \r", "\n\r", "\n"]), [["token", " spaced  "]]);
assert.deepEqual(parse(["event: token\ndata: partial"]), []);
assert.deepEqual(parse(["event: done\ndata: {}\n"]), []);
assert.deepEqual(parse(["event: token\n\ndata: value\n\n"]), [["message", "value"]]);
assert.deepEqual(parse(["data:\n\n"]), [["message", ""]]);
assert.equal(errorText({ quota: "daily_pages", remaining: 0 }), "quota: daily_pages · remaining: 0");

async function uiContracts() {
    const vm = require("node:vm"), fs = require("node:fs");
    const all = [], elements = new Map();
    let document;
    class Element {
        constructor(tag) {
            this.tag = tag; this.children = []; this.dataset = {}; this.style = {};
            this.handlers = {}; this.attributes = {}; this.value = ""; this.className = "";
            this.classList = { toggle() {} }; this.isConnected = true; all.push(this);
        }
        set textContent(text) { this.text = String(text); this.children = []; }
        get textContent() { return (this.text || "") + this.children.map(c => c.textContent).join(""); }
        append(...children) { for (const child of children) { child.remove(); child.parent = this; this.children.push(child); } }
        replaceChildren(...children) { this.children.forEach(c => { c.parent = null; }); this.children = []; this.text = ""; this.append(...children); }
        remove() { if (this.parent) this.parent.children = this.parent.children.filter(c => c !== this); this.parent = null; }
        replaceWith(other) { const parent = this.parent, index = parent.children.indexOf(this); other.remove(); parent.children[index] = other; other.parent = parent; this.parent = null; }
        setAttribute(key, value) { this.attributes[key] = value; }
        removeAttribute(key) { delete this.attributes[key]; }
        addEventListener(key, action) { this.handlers[key] = action; }
        querySelector(selector) {
            for (const child of this.children) {
                if (selector.startsWith(".") && child.className.split(" ").includes(selector.slice(1))) return child;
                const found = child.querySelector(selector); if (found) return found;
            }
            return null;
        }
        focus() { document.activeElement = this; }
    }
    document = {
        querySelector(selector) { if (!elements.has(selector)) elements.set(selector, new Element("div")); return elements.get(selector); },
        querySelectorAll() { return all.filter(el => Object.hasOwn(el.dataset, "navigation")); },
        createElement(tag) { return new Element(tag); },
        createTextNode(text) { const el = new Element("text"); el.textContent = text; return el; },
        documentElement: new Element("html"), body: new Element("body"),
    };
    const $ = selector => document.querySelector(selector);
    $("#modal").hidden = true;
    for (const id of ["#newChat", "#clearScope", "#versionMode"]) $(id).dataset.navigation = "";
    let establishSession;
    const session = new Promise(resolve => { establishSession = resolve; });
    const calls = [], payloads = [], storage = new Map();
    let askCount = 0, stream, failDocs = false, failUploadRefresh = false;
    const documents = [
        { id: "d1", filename: "report.pdf", version: 1, status: "ready" },
        { id: "d2", filename: "report.pdf", version: 2, status: "ready" },
    ];
    const implicit = { document_id: null, document_ids: null, version_mode: "latest" };
    const pinned = { document_id: "d1", document_ids: ["d1"], version_mode: "latest" };
    const empty = { ...implicit, document_ids: [] };
    const histories = {
        c1: [],
        latest: [{ role: "assistant", content: "old answer", meta: { requested_scope: implicit, scope: pinned, resolved_document_ids: ["d1"] } }],
        all: [{ role: "assistant", meta: { requested_scope: { ...implicit, version_mode: "all" }, resolved_document_ids: ["d1", "d2"] } }],
        pinned: [{ role: "assistant", meta: { requested_scope: pinned, resolved_document_ids: ["d2"] } }],
        empty: [{ role: "assistant", meta: { requested_scope: empty, resolved_document_ids: [] } }],
        legacy: [{ role: "assistant", meta: { scope: { ...implicit, document_ids: ["d1", "d2"], version_mode: "all" } } }],
        legacyImplicit: [{ role: "assistant", meta: { scope: pinned } }],
        bound: [{ role: "assistant", meta: { requested_scope: implicit, resolved_document_ids: ["d1"] } }],
        boundEmpty: [{ role: "assistant", meta: { requested_scope: empty } }],
        boundStale: [{ role: "assistant", meta: { scope: { ...pinned, document_id: "d2", document_ids: ["d2"] } } }],
    };
    const conversations = Object.keys(histories).map(id => ({ id, title: id,
        document_id: id.startsWith("bound") ? "d1" : null }));
    storage.set("vrag_scope_latest", JSON.stringify(pinned));
    storage.set("vrag_scope_pinned", JSON.stringify(implicit));
    storage.set("vrag_scope_legacyImplicit", JSON.stringify(implicit));
    const json = data => ({ ok: true, status: 200, json: async () => data });
    const mockFetch = async (path, options = {}) => {
        calls.push(path);
        if (path === "/api/session") { await session; return json({}); }
        if (path === "/readyz") return json({ ready: true });
        if (path === "/api/documents") {
            if (failDocs) { failDocs = false; throw new Error("offline document refresh"); }
            return json([...documents]);
        }
        if (path === "/api/conversations") return json(conversations);
        const history = path.match(/^\/api\/conversations\/([^/]+)\/messages$/);
        if (history) return json(histories[history[1]]);
        if (path === "/api/documents/upload") {
            if (!documents.some(d => d.id === "failed")) documents.push({ id: "failed", filename: "failed.pdf", version: 1, status: "error", last_error: "Queue unavailable" });
            failDocs = failUploadRefresh;
            return { ok: false, status: 503, headers: { get: () => null }, json: async () => ({ detail: "Queue unavailable" }) };
        }
        if (path === "/api/documents/failed/detail") return json({ document: documents.find(d => d.id === "failed"), jobs: [] });
        if (path === "/api/usage") return json({ day: "2026-09-24", timezone: "UTC", limits: {}, usage: {}, remaining: {} });
        if (path === "/api/ask/stream") {
            payloads.push(JSON.parse(options.body)); askCount++;
            const reader = {
                pending: null, queue: [], cancelled: false,
                async read() { return this.queue.length ? this.queue.shift() : new Promise(resolve => { this.pending = resolve; }); },
                push(text) { const chunk = { done: false, value: new TextEncoder().encode(text) }; if (this.pending) { const resolve = this.pending; this.pending = null; resolve(chunk); } else this.queue.push(chunk); },
                async cancel() { this.cancelled = true; if (this.pending) { this.pending({ done: true }); this.pending = null; } },
                releaseLock() {},
            };
            stream = reader;
            options.signal.addEventListener("abort", () => reader.cancel(), { once: true });
            return { ok: true, headers: { get: () => "text/event-stream" }, body: { getReader: () => reader } };
        }
        throw new Error(`Unexpected offline request: ${path}`);
    };
    const sandbox = { document, fetch: mockFetch, TextEncoder, TextDecoder, AbortController, DOMException,
        FormData, URL, URLSearchParams, Blob, console, setTimeout: () => 1, clearTimeout() {}, setInterval() {},
        location: { origin: "https://offline.invalid" },
        sessionStorage: { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
        localStorage: { getItem: () => null, setItem() {} },
    };
    vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
    assert.deepEqual(calls, ["/api/session"]);
    assert.equal($("#chat").children.length, 0);
    establishSession();
    const tick = () => new Promise(resolve => setImmediate(resolve));
    const checkboxFor = id => $("#docList").children.find(li => li.children[0]?.children[0]?.attributes["aria-label"] === `Search report.pdf version ${id === "d1" ? 1 : 2}`).children[0].children[0];
    const select = id => { const checkbox = checkboxFor(id); checkbox.checked = true; checkbox.onchange(); };
    const openChat = async id => {
        const entry = $("#convList").children.find(li => li.children[0]?.textContent === id);
        await entry.children[0].onclick();
    };
    const stored = id => JSON.parse(storage.get(`vrag_scope_${id}`));
    await tick();
    assert.equal($("#chat").children.length, 0, "boot must keep chat empty");
    select("d1"); select("d2");
    $("#prompt").value = "original question"; $("#sendBtn").onclick();
    $("#prompt").value = "must not submit while busy";
    $("#prompt").handlers.keydown({ key: "Enter", preventDefault() {} });
    assert.equal(askCount, 1, "Enter must respect busy guard");
    const count = $("#chat").children.length; $("#newChat").onclick();
    assert.equal($("#chat").children.length, count, "busy navigation must not clear the active request");
    await tick();
    stream.push('event: start\ndata: {"conversation_id":"c1"}\n\nevent: error\ndata: {"error":"offline failure"}\n\n');
    await tick();
    const retry = all.find(el => el.className === "retry-btn");
    assert.ok(retry, "failure must create a retry button dynamically");
    assert.equal(retry.disabled, false);
    $("#clearScope").onclick();
    const retried = retry.onclick(); await tick();
    assert.equal(askCount, 2);
    assert.equal(payloads[1].question, "original question");
    assert.deepEqual(payloads[1].document_ids, ["d1", "d2"], "retry must use captured scope, not changed global selection");
    assert.equal(payloads[1].conversation_id, "c1");
    assert.equal(retry.disabled, true, "retry controls lock while streaming");
    assert.equal($("#stopBtn").hidden, false);
    assert.equal($("#clearScope").disabled, true);
    await retry.onclick(); assert.equal(askCount, 2, "repeat retry clicks must not duplicate requests");
    const beforeDone = calls.filter(path => path === "/api/conversations").length;
    stream.push('event: done\ndata: {"answer":"Final [E1] <img src=x>","message_id":7,"conversation_id":"c1","status":"answered","confidence_available":false,"sources":[{"label":"E1","document_id":"d1","evidence_id":"e1","filename":"report.pdf","page":1}]}\n\n');
    await retried; await tick();
    assert.ok(calls.filter(path => path === "/api/conversations").length > beforeDone, "done refreshes sidebar even after start set conversation ID");
    assert.ok($("#chat").textContent.includes("Final [E1] <img src=x>"));
    assert.ok(all.some(el => el.className === "citation" && el.textContent === "[E1]"));
    assert.equal(all.filter(el => el.tag === "img").length, 0, "answer text must never create markup");
    $("#newChat").onclick();
    await retry.onclick(); assert.equal(askCount, 2, "old retries must not leak into a new conversation");
    await openChat("c1");
    assert.ok($("#scopeChip").textContent.includes("report.pdf"), "opening a conversation restores document scope");
    $("#prompt").value = "stop this"; $("#sendBtn").onclick(); await tick();
    stream.push('event: token\ndata: {"text":"unconfirmed text"}\n\n'); await tick();
    $("#stopBtn").onclick();
    assert.equal($("#stopBtn").disabled, true);
    assert.equal($("#sendBtn").disabled, true, "stop does not unlock before cleanup");
    await tick();
    assert.ok(stream.cancelled); assert.equal($("#sendBtn").disabled, false);
    assert.equal($("#clearScope").disabled, false);
    assert.equal($("#stopBtn").hidden, true);
    assert.ok($("#streamStatus").textContent.includes("Stopped"));
    assert.ok(!$("#chat").textContent.includes("unconfirmed text"), "cancelled partial output is not presented as confirmed");
    const stoppedStream = stream;
    $("#newChat").onclick();
    stoppedStream.push('event: done\ndata: {"answer":"late answer","message_id":99}\n\n'); await tick();
    assert.equal($("#chat").children.length, 0, "late cancelled frames cannot populate the new chat");

    // Metadata describes the request, not the versions that resolved last time.
    async function checkScope(id, expected, label) {
        await openChat(id);
        assert.deepEqual(stored(id), expected);
        assert.ok($("#scopeChip").textContent.includes(label));
        $("#prompt").value = `scope ${id}`; $("#sendBtn").onclick(); await tick();
        assert.deepEqual(payloads.at(-1), { question: `scope ${id}`, conversation_id: id, ...expected });
        $("#stopBtn").onclick(); await tick();
    }
    await checkScope("latest", implicit, "Latest ready version per filename");
    assert.equal(checkboxFor("d1").checked, false);
    assert.equal(checkboxFor("d2").checked, false);
    // A newer ready version must not convert an implicit request into a pin.
    documents.push({ id: "d3", filename: "report.pdf", version: 3, status: "ready" });
    await $("#refreshDocs").onclick();
    await checkScope("latest", implicit, "Latest ready version per filename");
    await checkScope("all", { ...implicit, version_mode: "all" }, "All ready versions");
    await checkScope("pinned", pinned, "report.pdf v1");
    await checkScope("empty", empty, "Empty exact scope");
    await checkScope("legacy", { ...implicit, document_ids: ["d1", "d2"], version_mode: "all" }, "Exact scope");
    await checkScope("legacyImplicit", implicit, "Latest ready version per filename");
    await checkScope("bound", implicit, "report.pdf v1");
    assert.ok($("#scopeChip").textContent.includes("Conversation bound"));
    await checkScope("boundEmpty", empty, "Empty exact scope");
    await checkScope("boundStale", pinned, "report.pdf v1");

    // Clear/multi-select cannot keep using a conversation bound on the server.
    await openChat("bound");
    $("#clearScope").onclick();
    assert.equal($("#chat").children.length, 0);
    assert.equal($("#crumb").textContent, "New chat");
    assert.ok(document.body.textContent.includes("Started a new chat"));
    $("#prompt").value = "new implicit chat"; $("#sendBtn").onclick(); await tick();
    assert.deepEqual(payloads.at(-1), { question: "new implicit chat", conversation_id: null, ...implicit });
    $("#stopBtn").onclick(); await tick();
    await openChat("bound"); select("d2");
    $("#prompt").value = "new multi chat"; $("#sendBtn").onclick(); await tick();
    assert.equal(payloads.at(-1).conversation_id, null);
    assert.deepEqual(payloads.at(-1).document_ids, ["d1", "d2"]);
    $("#stopBtn").onclick(); await tick();
    await openChat("bound");
    $("#versionMode").value = "all"; $("#versionMode").onchange({ target: $("#versionMode") });
    assert.ok($("#scopeChip").textContent.includes("report.pdf v1"), "version mode does not unpin a bound conversation");
    $("#prompt").value = "still bound"; $("#sendBtn").onclick(); await tick();
    assert.equal(payloads.at(-1).conversation_id, "bound");
    assert.equal(payloads.at(-1).version_mode, "all");
    $("#stopBtn").onclick(); await tick();

    // Failed registration/queueing must surface the document and its retry UI.
    const file = new Blob(["%PDF-offline"], { type: "application/pdf" }); file.name = "failed.pdf";
    const upload = async () => {
        const docsBefore = calls.filter(path => path === "/api/documents").length;
        const usageBefore = calls.filter(path => path === "/api/usage").length;
        $("#fileInput").onchange({ target: { files: [file], value: "file" } });
        assert.equal($("#fileInput").disabled, true);
        await tick();
        assert.ok(calls.filter(path => path === "/api/documents").length > docsBefore);
        assert.ok(calls.filter(path => path === "/api/usage").length > usageBefore);
        assert.equal($("#fileInput").disabled, false);
        assert.equal($("#dropzone").attributes["aria-busy"], "false");
        assert.ok($("#dzText").textContent.includes("Upload failed: Queue unavailable"));
    };
    await upload();
    const failedDoc = $("#docList").children.find(li => li.textContent.includes("failed.pdf"));
    assert.ok(failedDoc.textContent.includes("Queue unavailable"));
    await failedDoc.children[1].onclick();
    const retryIngestion = all.find(el => el.textContent === "Retry ingestion");
    assert.equal(retryIngestion.disabled, false);
    $("#modalClose").onclick();
    failUploadRefresh = true;
    await upload(); // A secondary refresh error must not mask the upload error or lock controls.
    console.log("Offline UI contracts passed (scope restoration/bindings, failed uploads, retry locking, cancellation, safe citations).");
}

async function main() {
  // Tiny reader stub: no network, browser, timers or third-party dependencies.
  function response(text, width = 1) {
    const bytes = new TextEncoder().encode(text);
    let offset = 0;
    const reader = {
      cancelled: false, released: false,
      async read() {
        if (offset >= bytes.length) return { done: true };
        const value = bytes.slice(offset, offset + width); offset += width;
        return { done: false, value };
      },
      async cancel() { this.cancelled = true; },
      releaseLock() { this.released = true; },
    };
    return { ok: true, headers: { get: () => "text/event-stream" }, body: { getReader: () => reader }, reader };
  }
  const events = [];
  const complete = response('event: stage\r\ndata: {"stage":"verifying"}\r\n\r\nevent: token\r\ndata: {"text":"你好 🌍"}\r\n\r\nevent: done\r\ndata: {"answer":"Final [E1]",\r\ndata: "sources":[{"label":"E1"}],"status":"answered"}\r\n\r\n');
  await consumeAskStream(complete, (type, data) => events.push([type, data]));
  assert.equal(events[1][1].text, "你好 🌍");
  assert.equal(events[2][1].answer, "Final [E1]");
  assert.equal(events[2][1].sources[0].label, "E1");
  assert.ok(complete.reader.cancelled && complete.reader.released);
  for (const text of ["", 'event: token\ndata: {"text":"partial"}\n\n', 'event: done\ndata: {"answer":"unterminated"}\n']) {
    await assert.rejects(consumeAskStream(response(text), () => {}), /before final confirmation/);
  }
  await assert.rejects(consumeAskStream(response('event: error\ndata: {"error":{"message":"quota exhausted"}}\n\n'), () => {}), /quota exhausted/);
  await assert.rejects(consumeAskStream(response('event: token\ndata: invalid\n\n'), () => {}), SyntaxError);
  const errorResponse = { ok: false, status: 429, headers: { get: () => "10" }, json: async () => ({ detail: { resource: "daily_asks", remaining: 0 } }) };
  await assert.rejects(consumeAskStream(errorResponse, () => {}), /Quota.*daily_asks.*remaining: 0.*Retry after 10s/);
  let count = 0;
  await consumeAskStream(response('event: done\ndata: {"answer":"final"}\n\nevent: token\ndata: {"text":"ignored"}\n\n', 1000), () => count++);
  assert.equal(count, 1);
  const aborted = new AbortController(); aborted.abort();
  const abortResponse = response('event: done\ndata: {}\n\n');
  await assert.rejects(consumeAskStream(abortResponse, () => {}, aborted.signal), { name: "AbortError" });
  assert.ok(abortResponse.reader.cancelled && abortResponse.reader.released);
  const midstream = new AbortController();
  await assert.rejects(consumeAskStream(response('event: token\ndata: {"text":"a"}\n\nevent: done\ndata: {}\n\n'), () => midstream.abort(), midstream.signal), { name: "AbortError" });
  console.log("Frontend parser contracts passed (all split positions, UTF-8, terminal done, errors, abort).");
    await uiContracts();
}
main().catch(error => { console.error(error); process.exitCode = 1; });
'''


class Markup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: dict[str, tuple[str, dict]] = {}
        self.duplicates: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids[attrs["id"]] = (tag, attrs)


class FrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = APP.read_text(encoding="utf-8")
        cls.html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        cls.css = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")

    def test_markup_and_accessibility_contract(self):
        markup = Markup()
        markup.feed(self.html)
        self.assertEqual(markup.duplicates, [])
        for identifier in re.findall(r'\$\("#([\w-]+)"\)', self.js):
            self.assertIn(identifier, markup.ids)
        self.assertEqual(markup.ids["streamStatus"][1]["aria-live"], "polite")
        self.assertIn('role="dialog"', self.html)
        self.assertIn('aria-modal="true"', self.html)
        self.assertIn('event.key === "Escape"', self.js)
        self.assertIn('event.key !== "Tab"', self.js)
        self.assertIn("modalFocus.focus()", self.js)
        self.assertRegex(self.css, r"\[hidden\]\s*\{\s*display:\s*none\s*!important")
        self.assertRegex(self.html, r'<section[^>]+id="chat"[^>]*></section>')
        self.assertNotIn("empty-state", self.js)

    def test_safe_evidence_and_metadata_contract(self):
        for forbidden in ("innerHTML", "insertAdjacentHTML", "image_urls", '"/images/'):
            self.assertNotIn(forbidden, self.js)
        for field in ("evidence_id", "page_image_url", "bbox", "quote", "preview", "cited",
                      "abstention_reason", "answerable", "citation_warnings", "calibration",
                      "confidence_available", "confidence_unavailable_reason", "retrieval_query"):
            self.assertIn(field, self.js)
        self.assertIn("document.createTextNode", self.js)
        self.assertIn("url.origin !== location.origin", self.js)
        self.assertIn("assistant.content = obj.answer", self.js)
        self.assertIn("...assistant.meta, ...obj", self.js)

    def test_session_precedes_concurrent_boot(self):
        boot = self.js[self.js.index("async function boot()") :]
        self.assertLess(boot.index('await api("/api/session")'), boot.index("Promise.allSettled"))
        self.assertIn("if (!state.ready || state.busy || state.navigating)", self.js)
        self.assertIn("state.controller?.abort()", self.js)
        self.assertIn("captured.view !== state.view", self.js)
        self.assertIn("captured.conversation_id !== state.conversationId", self.js)
        self.assertNotIn("pendingRetry", self.js)
        self.assertIn("restoreScope", self.js)

    def test_lifecycle_usage_and_comparison_contract(self):
        for fragment in ("/detail", "/evidence/", "/api/documents/compare?", "/api/usage",
                         "doc.job_id", "pages_without_content", "failed_page", "attempt_count",
                         "last_error", "next_offset", "text_truncated", "total_pages",
                         "version_mode", "document_ids", "daily_pages", "daily_asks"):
            self.assertIn(fragment, self.js)
        self.assertIn('["done", "error", "cancelled"]', self.js)
        self.assertIn("watcher.attempts++ >= 40", self.js)
        self.assertIn("state.uploading) return", self.js)
        self.assertIn("not a visual pixel diff", self.js)

    def test_cli_namespace_and_content_identity_without_api_imports(self):
        for relative in ("main.py", "scripts/ingest.py", "scripts/query.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            tree = ast.parse(source)
            self.assertNotIn("src.api", source)
            self.assertIn("sha256_file", source)
            for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
                if call.func.id in {"ask", "upsert_documents", "clear_namespace"}:
                    namespace = next((k.value for k in call.keywords if k.arg == "namespace"), None)
                    self.assertIsNotNone(namespace, relative)
                    self.assertEqual(ast.unparse(namespace), "settings.pinecone_namespace")
                if call.func.id in {"ask", "extract_documents"}:
                    self.assertIn("document_id", {k.arg for k in call.keywords}, relative)

    def test_preview_does_not_pretend_to_serve_data(self):
        tree = ast.parse((ROOT / "scripts" / "serve_ui.py").read_text(encoding="utf-8"))
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(ast.literal_eval(functions["session"].body[-1].value), {})
        ready = ast.literal_eval(functions["readiness"].body[-1].value)
        self.assertIs(ready["ready"], False)
        unavailable = functions["backend_unavailable"]
        raised = next(n for n in unavailable.body if isinstance(n, ast.Raise))
        keywords = {k.arg: ast.literal_eval(k.value) for k in raised.exc.keywords}
        self.assertEqual(keywords["status_code"], 503)
        self.assertIn("UI-only preview", keywords["detail"])
        route = unavailable.decorator_list[0]
        self.assertEqual(ast.literal_eval(route.args[0]), "/api/{path:path}")
        self.assertNotIn("src.api", ast.unparse(tree))

    @unittest.skipUnless(shutil.which("node"), "Node is optional; static contracts still run")
    def test_isolated_sse_parser(self):
        result = subprocess.run(
            [shutil.which("node"), "-e", NODE_TESTS, str(APP)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()