Paper build instructions for the course project

To compile the LaTeX report (requires a LaTeX distribution):

```powershell
py -3 -m pip install -r requirements.txt
cd paper
pdflatex thesis_main.tex
bibtex thesis_main
pdflatex thesis_main.tex
pdflatex thesis_main.tex
```

Legacy (non-template) build (optional):

```powershell
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Notes:
- Figures are copied automatically from `results/` into `paper/figures/` by `scripts/run_all.py`.
- If `pdflatex` is not available on your PATH, install TeX Live or MikTeX and ensure `pdflatex` and `bibtex` are on PATH.
