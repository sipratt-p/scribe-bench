#!/usr/bin/env bash
# Build the paper: figures via headless Chrome (Mac), markdown -> LaTeX -> PDF via pandoc + pdflatex (beast).
set -e
cd "$(dirname "$0")/.."
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for f in fig1_decoder fig2_loop fig3_live; do
  python3 - "$f" <<'PY'
import re,sys
name=sys.argv[1]
vals={"--ink":"#172029","--muted":"#5C6B78","--line":"#CAD3DB","--accent":"#0F6B8A","--accent-soft":"#DDEEF4","--surface":"#FFFFFF","--bg":"#FFFFFF","--theirs":"#8A5A0F","--theirs-soft":"#F5EBD8","--ok":"#1E7B4F","--ok-soft":"#DDF1E6","--bad":"#B3261E","--bad-soft":"#F9E1DF","--live":"#8A5A0F","--live-soft":"#F5EBD8","--warn":"#8A5A0F","--warn-soft":"#F5EBD8"}
svg=open(f"wiki/{name}.svg").read()
svg=re.sub(r"var\((--[a-z-]+)\)", lambda m: vals.get(m.group(1),"#172029"), svg)
vb=re.search(r'viewBox="0 0 (\d+) (\d+)"', svg); w,h=int(vb.group(1)),int(vb.group(2))
open(f"paper/figs/{name}.html","w").write(f'<!doctype html><html><head><meta charset="utf-8"><style>@page{{size:{w}px {h}px;margin:0}} html,body{{margin:0;background:#fff;color:#172029;width:{w}px;height:{h}px}} svg{{display:block;width:{w}px;height:{h}px}}</style></head><body>{svg}</body></html>')
PY
  "$CH" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/paper/figs/$f.pdf" "file://$PWD/paper/figs/$f.html" >/dev/null 2>&1
done
uv run python paper/md2tex.py
rsync -aq paper/ beast:~/projects/scribe-bench/paper/
ssh -o BatchMode=yes beast 'cd ~/projects/scribe-bench/paper && pandoc paper.md -o main.tex --template template.tex --from markdown+pipe_tables --wrap=none --shift-heading-level-by=-1 && xelatex -interaction=nonstopmode main.tex >build.log 2>&1; xelatex -interaction=nonstopmode main.tex >build.log 2>&1; pdfinfo main.pdf | grep Pages'
scp -q beast:~/projects/scribe-bench/paper/main.pdf paper/main.pdf
echo "paper/main.pdf built"
