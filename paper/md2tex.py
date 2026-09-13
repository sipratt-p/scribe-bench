"""Turn wiki/paper-draft.md into paper/paper.md (pandoc-ready): drop the status line, swap inline-SVG figure blocks for PDF includes, make the reference list a plain numbered list. Then: pandoc paper/paper.md -o paper/main.tex --standalone --template paper/template.tex"""
import re
from pathlib import Path
s = Path("wiki/paper-draft.md").read_text()
s = re.sub(r"\*\*Status:\*\*.*?\n\n---\n\n", "", s, flags=re.S)
s = s.replace("# Paper draft (workshop): What an LLM-judged evaluation of ambient clinical scribes can and cannot see\n\n", "")
# figures: replace <div style="overflow-x:auto"><svg ...>...</svg></div> with image includes, keyed by the heading just above
figs = {"## Figure 1.": "fig1_decoder", "## Figure 2.": "fig2_loop", "## Figure 3.": "fig3_live"}
def repl(m):
    head = m.group(1)
    key = next((v for k, v in figs.items() if head.startswith(k)), None)
    cap = head.split(". ", 1)[1] if ". " in head else head
    return f"![{cap}](figs/{key}.pdf){{width=100%}}\n" if key else ""
s = re.sub(r"(## Figure \d\. [^\n]+)\n<div style=\"overflow-x:auto\">.*?</div>\n", repl, s, flags=re.S)
s = s.replace("[[sources]]", "the repository").replace("[[live-experiments]]", "the run log").replace("[[autoresearch-loop]]", "the search log").replace("[[verifier]]", "Appendix C").replace("[[fairness]]", "Appendix D")
s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)
s = s.replace("## Title\n", "")
s = s.replace("## Abstract (≈200 words)", "## Abstract")
s = s.replace("## 2a. Related work (every reference below verified against the arXiv, ACL Anthology, PMC, Europe PMC or medRxiv record on 13 Sep 2026)", "## 2a. Related work")
s = s.replace("### References (verified)", "### References")
s = s.replace("## 6a. Data licensing and ethics", "## 6a. Data licensing and ethics")
s = re.sub(r"\*\*What an LLM-judged evaluation of ambient clinical scribes can and cannot see: an open harness and two case studies\*\*\n\nAlternative:[^\n]*\n", "", s)
Path("paper/paper.md").write_text(s)
print("paper/paper.md written", len(s))
