# Paper — IEEE HPEC 2026 Camera-Ready Manuscript

LaTeX source for the accepted IEEE HPEC 2026 paper.

## Contents

| File | Description |
|------|-------------|
| `paper.tex` | Main LaTeX manuscript (6 sections: Introduction, Background, Methodology, Results, Discussion, Conclusion) |
| `references.bib` | BibTeX bibliography |
| [`../requirements.txt`](../requirements.txt) | Maintained dependencies for new runs; see the versioning note below |
| `figures/` | Publication figures (PNG + SVG, IEEE format) |
| `paper.pdf` | Compiled PDF (camera-ready) |
| `IEEEtran.cls` | IEEE conference template class file |
| `CAMERA_READY_CHECKLIST.md` | Verified requirements, exact CMT metadata, and remaining account-dependent submission steps |

The published measurements use OpenVINO 2024.1 and ONNX Runtime 1.18. The root `requirements.txt` tracks a newer maintained environment, so results produced with it are new measurements rather than a bit-for-bit reproduction of the paper.

## Structure

The paper follows standard IEEE conference format:
- **Title:** Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis
- **Sections:** Abstract → Introduction → Background and Related Work → Methodology → Results → Discussion → Conclusion
- **7 figures:** Latency comparison, INT8 speedup heatmap, operator composition, optimization speedup, roofline analysis, density analysis, and scaling analysis
- **5 tables:** Model inventory, dataset characteristics, NPU INT8 performance, device-assignment exceptions, and package power

The earlier CPU-fallback heatmap is retained as a diagnostic artifact in `results/figures/` but is not included in the camera-ready manuscript because its nearly uniform values obscured the small set of meaningful exceptions.

## Compilation

Regenerate the crowded single-column vector figures at their final IEEE print size before compiling:

```bash
python analysis/generate_ieee_paper_figures.py
```

This produces matching PNG, SVG, and vector PDF files for paper Figures 1-7. The manuscript embeds the PDF variants so 8 pt axis labels are preserved at one-column width.

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```
