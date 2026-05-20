# Paper — Ieee Conference Paper

Latex source for the ieee conference paper submitted for publication.

## Contents

| File | Description |
|------|-------------|
| `paper.tex` | Main LaTeX manuscript (6 sections: Introduction, Background, Methodology, Results, Discussion, Conclusion) |
| `references.bib` | BibTeX bibliography |
| `figures/` | Publication figures (PNG + SVG, IEEE format) |
| `paper.pdf` | Compiled PDF (camera-ready) |
| `IEEEtran.cls` | IEEE conference template class file |

## Structure

The paper follows standard IEEE conference format:
- **Title:** Benchmarking GNN Inference Bottlenecks on Intel Core Ultra NPUs
- **Sections:** Abstract → Introduction → Background and Related Work → Methodology → Results → Discussion → Conclusion
- **7 figures:** Latency comparison, INT8 speedup heatmap, operator breakdown, CPU fallback heatmap, optimization speedup, roofline analysis, density vs. latency

## Compilation

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```
