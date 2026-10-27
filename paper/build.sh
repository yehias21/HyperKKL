#!/usr/bin/env bash
# Build the TMLR submission. Run from anywhere: bash paper/build.sh
# Re-run after every regeneration of paper/generated (numbers.tex, tab*.tex, fig*.pdf).
set -e
cd "$(dirname "$0")"
if [ -f ../experiments/real/results.json ]; then ../.venv/bin/python -I ../analysis/real_numbers.py; fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
echo "pages: $(pdfinfo main.pdf | awk '/Pages/{print $2}')"
echo "undefined references/citations: $(grep -c 'undefined' build.log || true)"
echo "multiply-defined labels: $(grep -c 'multiply defined' build.log || true)"
# main-text page count = page on which the References heading sits, minus one if it is at the top
for p in $(seq 1 30); do
  if pdftotext -f $p -l $p -layout main.pdf - 2>/dev/null | grep -q '^References'; then
    n=$(pdftotext -f $p -l $p -layout main.pdf - | grep -v '^\s*$' | grep -n '^References' | cut -d: -f1)
    if [ "$n" -le 2 ]; then echo "main text ends on page $((p-1))"; else echo "main text ends on page $p (References start mid-page)"; fi
    break
  fi
done
# visible placeholders mean a macro vanished from numbers.tex
pdftotext -layout main.pdf - | grep -n '\[pending:' || echo "no missing-macro placeholders"
