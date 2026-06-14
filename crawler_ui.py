"""
crawler_ui.py — a small local web UI for pdf_crawler.py.

A zero-dependency control panel (Python standard library only — no Flask, no
extra installs beyond what pdf_crawler.py already needs) for configuring and
running the AIM Composites source-discovery crawler from the browser:

  * pick queries, output dir, per-query / total caps, min relevance score,
    and the skip-datasheets / skip-academic toggles;
  * launch the crawl as a subprocess and watch its log stream live;
  * stop a running crawl with a button (real process termination);
  * browse the resulting sources.csv as a table, opening each downloaded
    PDF straight from disk.

It shells out to `python pdf_crawler.py ...` rather than importing it, so a
crawl can be cancelled cleanly and a crash in the crawler can't take the UI
down with it.

Usage:
    python crawler_ui.py                 # serves on http://127.0.0.1:8765
    python crawler_ui.py --port 9000
    python crawler_ui.py --open          # also open the browser automatically

Binds to 127.0.0.1 only — this is a local tool, not a public server.

Author: Mathias Heider, AIM Composites project, June 2026.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRAWLER = HERE / "pdf_crawler.py"


# ---------------------------------------------------------------------------
# Crawl process manager — one crawl at a time, with a live log buffer
# ---------------------------------------------------------------------------

class CrawlManager:
    """Owns at most one running crawl subprocess and its captured log lines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._lines: list[str] = []
        self._out_dir: Path | None = None
        self._returncode: int | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def out_dir(self) -> Path | None:
        with self._lock:
            return self._out_dir

    def start(self, opts: dict) -> tuple[bool, str]:
        if self.running:
            return False, "A crawl is already running."
        if not CRAWLER.exists():
            return False, f"pdf_crawler.py not found next to this UI ({CRAWLER})."

        out_dir = Path(opts.get("out") or "./crawl_out")
        argv = self._build_argv(opts, out_dir)

        with self._lock:
            self._lines = [f"$ {' '.join(_shquote(a) for a in argv)}", ""]
            self._out_dir = out_dir
            self._returncode = None
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            try:
                self._proc = subprocess.Popen(
                    argv, cwd=str(HERE), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
            except OSError as e:
                self._proc = None
                return False, f"Failed to launch crawler: {e}"
            proc = self._proc

        threading.Thread(target=self._pump, args=(proc,), daemon=True).start()
        return True, "started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return False, "No crawl is running."
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        self._append("", "[stopped by user]")
        return True, "stopped"

    def status(self, since: int) -> dict:
        with self._lock:
            total = len(self._lines)
            new = self._lines[max(0, since):]
            return {
                "running": self._proc is not None and self._proc.poll() is None,
                "lines": new,
                "total": total,
                "returncode": self._returncode,
                "out": str(self._out_dir) if self._out_dir else None,
            }

    # -- internals ----------------------------------------------------------

    def _build_argv(self, opts: dict, out_dir: Path) -> list[str]:
        argv = [sys.executable, "-u", str(CRAWLER), "--out", str(out_dir)]
        queries = [q.strip() for q in (opts.get("queries") or "").splitlines()
                   if q.strip()]
        if queries:
            argv += ["--queries", *queries]
        seeds = [s.strip() for s in (opts.get("seed_urls") or "").splitlines()
                 if s.strip()]
        if seeds:
            argv += ["--seed-urls", *seeds]
        if opts.get("max_per_query"):
            argv += ["--max-per-query", str(int(opts["max_per_query"]))]
        if opts.get("max_total"):
            argv += ["--max-total", str(int(opts["max_total"]))]
        if opts.get("min_score"):
            argv += ["--min-score", str(int(opts["min_score"]))]
        if opts.get("email"):
            argv += ["--email", str(opts["email"]).strip()]
        if opts.get("skip_datasheets"):
            argv.append("--skip-datasheets")
        if opts.get("skip_academic"):
            argv.append("--skip-academic")
        if opts.get("verbose"):
            argv.append("--verbose")
        return argv

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self._append(line.rstrip("\n"))
        proc.stdout.close()
        rc = proc.wait()
        with self._lock:
            self._returncode = rc
        self._append("", f"[crawler exited with code {rc}]")

    def _append(self, *lines: str) -> None:
        with self._lock:
            self._lines.extend(lines)


def _shquote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or not arg) else arg


MANAGER = CrawlManager()


# ---------------------------------------------------------------------------
# Reading crawl results (sources.csv) for the results table
# ---------------------------------------------------------------------------

def read_sources(out_dir: Path) -> list[dict]:
    csv_path = out_dir / "sources.csv"
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    return rows


def safe_pdf_path(out_dir: Path, name: str) -> Path | None:
    """Resolve a PDF filename to a path strictly inside <out_dir>/pdfs."""
    pdf_dir = (out_dir / "pdfs").resolve()
    target = (pdf_dir / name).resolve()
    if pdf_dir == target or pdf_dir in target.parents:
        if target.is_file() and target.suffix.lower() == ".pdf":
            return target
    return None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "AIMCrawlerUI/1.0"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- routing ------------------------------------------------------------

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(url.query)
        if url.path == "/":
            self._send_html(PAGE)
        elif url.path == "/status":
            since = int((qs.get("since") or ["0"])[0] or 0)
            self._send_json(MANAGER.status(since))
        elif url.path == "/results":
            out = MANAGER.out_dir()
            rows = read_sources(out) if out else []
            self._send_json({"out": str(out) if out else None, "rows": rows})
        elif url.path == "/pdf":
            self._serve_pdf(qs)
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/start":
            opts = self._read_json()
            ok, msg = MANAGER.start(opts or {})
            self._send_json({"ok": ok, "message": msg}, code=200 if ok else 409)
        elif url.path == "/stop":
            ok, msg = MANAGER.stop()
            self._send_json({"ok": ok, "message": msg}, code=200 if ok else 409)
        else:
            self.send_error(404, "Not found")

    # -- helpers ------------------------------------------------------------

    def _serve_pdf(self, qs: dict):
        out = MANAGER.out_dir()
        name = (qs.get("name") or [""])[0]
        if out is None or not name:
            self.send_error(404, "No such PDF")
            return
        path = safe_pdf_path(out, name)
        if path is None:
            self.send_error(404, "No such PDF")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _send_json(self, obj, code: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text: str):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Front-end (single self-contained page)
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIM Composites — PDF Crawler</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --fg:#e6e9ef;
          --muted:#9aa3b2; --accent:#4f8cff; --good:#3ecf8e; --bad:#ff6b6b; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:12px; }
  header h1 { font-size:17px; margin:0; font-weight:600; }
  .dot { width:10px; height:10px; border-radius:50%; background:var(--muted); }
  .dot.run { background:var(--good); box-shadow:0 0 8px var(--good); }
  .dot.err { background:var(--bad); }
  main { display:grid; grid-template-columns:340px 1fr; gap:0;
         height:calc(100vh - 59px); }
  .config { padding:20px 24px; border-right:1px solid var(--line);
            overflow:auto; }
  .right { display:flex; flex-direction:column; min-width:0; }
  label { display:block; margin:14px 0 4px; color:var(--muted); font-size:12px;
          text-transform:uppercase; letter-spacing:.04em; }
  input[type=text], input[type=number], textarea {
    width:100%; background:#0c0e13; border:1px solid var(--line); color:var(--fg);
    border-radius:7px; padding:8px 10px; font:inherit; }
  textarea { resize:vertical; min-height:64px; font-family:ui-monospace,monospace;
             font-size:12px; }
  .row { display:flex; gap:10px; }
  .row > div { flex:1; }
  .checks { margin-top:12px; display:flex; flex-direction:column; gap:8px; }
  .checks label { display:flex; align-items:center; gap:8px; margin:0;
                  text-transform:none; letter-spacing:0; font-size:13px;
                  color:var(--fg); cursor:pointer; }
  .checks input { width:auto; }
  .btns { display:flex; gap:10px; margin-top:20px; }
  button { flex:1; padding:10px; border:0; border-radius:8px; font:inherit;
           font-weight:600; cursor:pointer; }
  #run { background:var(--accent); color:#fff; }
  #stop { background:#2a2f3a; color:var(--fg); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .tabs { display:flex; gap:4px; padding:10px 16px 0; border-bottom:1px solid var(--line); }
  .tab { padding:8px 14px; border-radius:8px 8px 0 0; cursor:pointer;
         color:var(--muted); }
  .tab.active { background:var(--panel); color:var(--fg); }
  .pane { flex:1; overflow:auto; background:var(--panel); padding:16px; min-height:0; }
  #log { white-space:pre-wrap; font-family:ui-monospace,monospace; font-size:12.5px;
         margin:0; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
           vertical-align:top; }
  th { color:var(--muted); font-weight:600; position:sticky; top:0;
       background:var(--panel); }
  td a { color:var(--accent); text-decoration:none; }
  td a:hover { text-decoration:underline; }
  .src { font-size:11px; padding:2px 7px; border-radius:99px; background:#222836;
         color:var(--muted); }
  .stat { color:var(--muted); padding:6px 16px; font-size:12px; }
  .empty { color:var(--muted); padding:40px; text-align:center; }
</style>
</head>
<body>
<header>
  <span id="dot" class="dot"></span>
  <h1>AIM Composites · PDF Crawler</h1>
  <span id="state" class="stat">idle</span>
</header>
<main>
  <section class="config">
    <label>Queries (one per line — blank = built-in set)</label>
    <textarea id="queries" placeholder="carbon fiber PEEK composite tensile&#10;PPS glass fiber mechanical"></textarea>

    <label>Output folder</label>
    <input type="text" id="out" value="./crawl_out">

    <div class="row">
      <div><label>Max / query</label><input type="number" id="max_per_query" value="8" min="1"></div>
      <div><label>Max total</label><input type="number" id="max_total" placeholder="none" min="1"></div>
      <div><label>Min score</label><input type="number" id="min_score" value="4" min="0"></div>
    </div>

    <label>Contact email (optional)</label>
    <input type="text" id="email" placeholder="you@example.com">

    <label>Datasheet seed URLs (one per line, optional)</label>
    <textarea id="seed_urls" placeholder="leave blank for built-in vendor list"></textarea>

    <div class="checks">
      <label><input type="checkbox" id="skip_datasheets"> Skip manufacturer datasheet crawl</label>
      <label><input type="checkbox" id="skip_academic"> Skip academic sources (OpenAlex/S2/arXiv)</label>
      <label><input type="checkbox" id="verbose"> Verbose (debug) logging</label>
    </div>

    <div class="btns">
      <button id="run">Start crawl</button>
      <button id="stop" disabled>Stop</button>
    </div>
  </section>

  <section class="right">
    <div class="tabs">
      <div class="tab active" data-pane="log">Live log</div>
      <div class="tab" data-pane="results">Results <span id="rcount"></span></div>
    </div>
    <div class="pane" id="pane-log"><pre id="log"></pre></div>
    <div class="pane" id="pane-results" style="display:none">
      <div id="results" class="empty">No results yet — start a crawl.</div>
    </div>
  </section>
</main>

<script>
let since = 0, polling = false, lastOut = null;
const $ = id => document.getElementById(id);

function setState(running, rc) {
  const dot = $('dot'), state = $('state');
  $('run').disabled = running;
  $('stop').disabled = !running;
  if (running) { dot.className = 'dot run'; state.textContent = 'crawling…'; }
  else if (rc && rc !== 0) { dot.className = 'dot err'; state.textContent = 'exited ('+rc+')'; }
  else { dot.className = 'dot'; state.textContent = 'idle'; }
}

document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  const p = t.dataset.pane;
  $('pane-log').style.display = p === 'log' ? '' : 'none';
  $('pane-results').style.display = p === 'results' ? '' : 'none';
  if (p === 'results') loadResults();
});

$('run').onclick = async () => {
  const opts = {
    queries: $('queries').value, out: $('out').value,
    seed_urls: $('seed_urls').value, email: $('email').value,
    max_per_query: $('max_per_query').value, max_total: $('max_total').value,
    min_score: $('min_score').value,
    skip_datasheets: $('skip_datasheets').checked,
    skip_academic: $('skip_academic').checked,
    verbose: $('verbose').checked,
  };
  const r = await fetch('/start', {method:'POST', body: JSON.stringify(opts)});
  const j = await r.json();
  if (!j.ok) { alert(j.message); return; }
  since = 0; $('log').textContent = '';
  setState(true, null);
  if (!polling) poll();
};

$('stop').onclick = async () => {
  await fetch('/stop', {method:'POST'});
};

async function poll() {
  polling = true;
  const r = await fetch('/status?since=' + since);
  const j = await r.json();
  if (j.lines.length) {
    since = j.total;
    const log = $('log');
    log.textContent += (log.textContent ? '\\n' : '') + j.lines.join('\\n');
    $('pane-log').scrollTop = $('pane-log').scrollHeight;
  }
  lastOut = j.out;
  setState(j.running, j.returncode);
  if (j.running) { setTimeout(poll, 1000); }
  else { polling = false; loadResults(); }
}

async function loadResults() {
  const r = await fetch('/results');
  const j = await r.json();
  const box = $('results');
  if (!j.rows || !j.rows.length) {
    box.className = 'empty';
    box.textContent = 'No results yet — start a crawl.';
    $('rcount').textContent = '';
    return;
  }
  $('rcount').textContent = '(' + j.rows.length + ')';
  box.className = '';
  let h = '<table><thead><tr><th>Title</th><th>Source</th><th>Year</th>' +
          '<th>Size</th><th>PDF</th><th>Origin</th></tr></thead><tbody>';
  for (const row of j.rows) {
    const kb = row.bytes ? (Number(row.bytes)/1024).toFixed(0) + ' kB' : '';
    const name = encodeURIComponent(row.filename || '');
    h += '<tr><td>' + esc(row.title) + '</td>' +
         '<td><span class="src">' + esc(row.source) + '</span></td>' +
         '<td>' + esc(row.year) + '</td><td>' + kb + '</td>' +
         '<td><a href="/pdf?name=' + name + '" target="_blank">open</a></td>' +
         '<td><a href="' + esc(row.url) + '" target="_blank">link</a></td></tr>';
  }
  box.innerHTML = h + '</tbody></table>';
}

function esc(s) {
  return (s || '').replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

// resume view if a crawl is already running when the page loads
poll();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"AIM Composites crawler UI -> {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        if MANAGER.running:
            MANAGER.stop()
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
