"""Render wiki/*.md into one browsable HTML page (sidebar + in-page anchors for wikilinks).
  uv run --with markdown python wiki/build_html.py <out.html>"""
import re, sys, html
from pathlib import Path
import markdown
W = Path(__file__).parent
order = ["index","glossary","overview","interview-narrative","log","decoder-finding","role-mapping","citations-and-verifiability",
         "fine-tuning-findings","verifier","plan-recall-gap","vanilla-vs-best","fairness","metrics","judges","autoresearch-loop",
         "expert-iteration","benchmark-maxing-vs-quality","two-pass-asr","models","datasets","infrastructure","abridge",
         "nvidia-engagement","deliverables","live-synthesis","live-experiments","paper-draft","open-questions","sources","README"]
pages = {p.stem: p.read_text() for p in W.glob("*.md")}
order += [k for k in sorted(pages) if k not in order]
def title(md): 
    m = re.search(r"^# (.+)$", md, re.M); return m.group(1) if m else "untitled"
def render(md):
    md = re.sub(r"\[\[([^\]|#]+)\]\]", lambda m: f'<a class="wl" href="#{m.group(1).strip()}">{html.escape(m.group(1).strip())}</a>', md)
    return markdown.markdown(md, extensions=["tables", "fenced_code"])
nav = "".join(f'<a href="#{k}">{html.escape(title(pages[k]))}</a>' for k in order if k in pages)
body = "".join(f'<section id="{k}"><div class="crumb">{k}</div>{render(pages[k])}</section>' for k in order if k in pages)
css = """
<title>Scribe Bench Wiki</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Public+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#F4F6F8;--surface:#FFFFFF;--ink:#172029;--muted:#5C6B78;--line:#CAD3DB;--accent:#0F6B8A;--accent-soft:#DDEEF4;--serif:Newsreader,Georgia,serif;--sans:'Public Sans',-apple-system,Helvetica,Arial,sans-serif;--mono:'IBM Plex Mono',Menlo,monospace;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0F151B;--surface:#161E26;--ink:#E4EAF0;--muted:#94A3B1;--line:#2E3A46;--accent:#52B7D6;--accent-soft:#12303B;color-scheme:dark}}
:root[data-theme=dark]{--bg:#0F151B;--surface:#161E26;--ink:#E4EAF0;--muted:#94A3B1;--line:#2E3A46;--accent:#52B7D6;--accent-soft:#12303B;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.55;margin:0}
.wrap{display:grid;grid-template-columns:260px minmax(0,1fr);min-height:100vh}
nav{position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid var(--line);background:var(--surface);padding:18px 14px;box-sizing:border-box}
nav .t{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 6px 10px}
nav a{display:block;padding:5px 8px;border-radius:4px;color:var(--ink);text-decoration:none;font-size:13.5px}
nav a:hover{background:var(--accent-soft)}
main{padding:28px 40px 80px;max-width:82ch;box-sizing:border-box}
section{padding:26px 0 30px;border-bottom:1px solid var(--line)}
.crumb{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.06em}
h1{font-family:var(--serif);font-weight:500;font-size:30px;line-height:1.15;margin:4px 0 12px;text-wrap:balance}
h2{font-family:var(--serif);font-weight:500;font-size:21px;margin:22px 0 8px}
h3{font-size:15px;margin:18px 0 6px}
p,li{max-width:72ch}
a{color:var(--accent)} a.wl{border-bottom:1px dotted var(--accent);text-decoration:none}
code{font-family:var(--mono);font-size:.9em;background:var(--accent-soft);padding:1px 4px;border-radius:3px}
pre{background:var(--surface);border:1px solid var(--line);padding:10px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;display:block;overflow-x:auto;font-variant-numeric:tabular-nums}
th,td{border-top:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
strong{font-weight:600}
@media (max-width:760px){.wrap{grid-template-columns:1fr}nav{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}main{padding:20px 16px 60px}}
</style>"""
out = css + f'<div class="wrap"><nav><div class="t">scribe-bench wiki</div>{nav}</nav><main>{body}</main></div>'
Path(sys.argv[1]).write_text(out); print("wrote", sys.argv[1], len(out))
