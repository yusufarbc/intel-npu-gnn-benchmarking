# Paper — IEEE HPEC 2026 Camera-Ready Manuscript

LaTeX source for the accepted IEEE HPEC 2026 paper.

## Contents

| File | Description |
|------|-------------|
| `paper.tex` | Main LaTeX manuscript (6 sections: Introduction, Background, Methodology, Results, Discussion, Conclusion) |
| `references.bib` | BibTeX bibliography |
| `requirements-core.txt` | Core OpenVINO and ONNX Runtime versions reported in the paper |
| `figures/` | Publication figures (PNG + SVG, IEEE format) |
| `paper.pdf` | Compiled PDF (camera-ready) |
| `IEEEtran.cls` | IEEE conference template class file |

## Structure

The paper follows standard IEEE conference format:
- **Title:** Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis
- **Sections:** Abstract → Introduction → Background and Related Work → Methodology → Results → Discussion → Conclusion
- **7 figures:** Latency comparison, INT8 speedup heatmap, operator composition, optimization speedup, roofline analysis, density analysis, and scaling analysis
- **5 tables:** Model inventory, dataset characteristics, NPU INT8 performance, device-assignment exceptions, and package power

The earlier CPU-fallback heatmap is retained as a diagnostic artifact in `results/figures/` but is not included in the camera-ready manuscript because its nearly uniform values obscured the small set of meaningful exceptions.

## Compilation

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```
