# Technical report build

Editable source and compiled PDF for the Phase 5 English technical report.

## Outputs

| Path | Role |
|------|------|
| `vlm_doc_technical_report.tex` | Editable LaTeX source |
| `references.bib` | Bibliography |
| `vlm_doc_technical_report.pdf` | Compiled report |
| `figures/` | PDF/SVG figures used by the report (generated from frozen results) |

## Report-build environment (separate from pinned ML env)

Do **not** install report tools into `~/.conda/envs/vlm_doc`.

```bash
cd /path/to/vlm_doc
python3 -m venv .venv_report
.venv_report/bin/pip install cairosvg pillow
```

System packages used for compile/preview on Neumann: TeX Live (`pdflatex`, `bibtex`), `mutool` (page renders).

## Regenerate figure PDFs from SVG (optional)

```bash
export PYTHONNOUSERSITE=1
.venv_report/bin/python - <<'PY'
from pathlib import Path
import cairosvg
for p in Path('docs/report/figures').glob('*.svg'):
    cairosvg.svg2pdf(url=str(p), write_to=str(Path('reports/figures')/(p.stem+'.pdf')))
PY
# gallery thumbs
mkdir -p reports/figures/gallery
cp docs/report/gallery_examples/*.jpg reports/figures/gallery/
```

Figures are also kept under `docs/report/figures/` for the evidence index.

## Compile PDF

```bash
cd reports
pdflatex -interaction=nonstopmode vlm_doc_technical_report.tex
bibtex vlm_doc_technical_report
pdflatex -interaction=nonstopmode vlm_doc_technical_report.tex
pdflatex -interaction=nonstopmode vlm_doc_technical_report.tex
```

Output: `reports/vlm_doc_technical_report.pdf`.

## Visual page preview (optional)

```bash
cd reports
mkdir -p page_previews
mutool draw -o page_previews/page-%02d.png -r 120 vlm_doc_technical_report.pdf
```

## Numeric cross-check

Frozen structured results live under `results/`. A Phase 5 check artifact is written to `results/phase05_report_numeric_check.json` when the packaging script/checks are run.
