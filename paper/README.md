# Paper - IEEE HPEC 2026 Camera-Ready Manuscript

LaTeX source for the accepted IEEE HPEC 2026 paper.

## Contents

| File | Description |
|------|-------------|
| `paper.tex` | Main LaTeX manuscript |
| `references.bib` | BibTeX bibliography |
| [`../requirements.txt`](../requirements.txt) | Maintained dependencies for new runs; see the versioning note below |
| `figures/` | The five vector PDF figures embedded by LaTeX |
| `paper.pdf` | Latest compiled manuscript PDF |
| `IEEEtran.cls` | IEEE conference template class file from `Conference-LaTeX-template_10-17-19` |
| `CAMERA_READY_CHECKLIST.md` | Final local and account-dependent submission checks |

The published measurements use OpenVINO 2024.1 and ONNX Runtime 1.18. The root `requirements.txt` tracks a newer maintained environment, so results produced with it are new measurements rather than a bit-for-bit reproduction of the paper.

The manuscript follows the IEEE `Conference-LaTeX-template_10-17-19` structure. Its local `IEEEtran.cls` is content-identical to the class distributed with that package (line endings may differ). The source does not override the template's margins, column widths, font sizes, or float spacing.

## Evidence boundaries

- GNN measurements use fixed 2,708-node, 10,000-edge inputs derived from three OGB source graphs; they are not full-graph OGB measurements.
- Dense baselines use the same synthetic input under each dataset label.
- Requested-device labels and registered providers do not alone prove per-operator device placement.
- Package power is measured over a benchmark window. The paper's latency-derived energy values are heuristic estimates, not direct active-inference energy measurements.
- The camera-ready paper excludes the exploratory density-correlation and heuristic roofline plots because the retained measurements do not support those interpretations.

## Figures

The camera-ready manuscript contains five figures: FP32 latency, INT8 speedup, ONNX operator composition, graph-optimization sensitivity, and APPNP fixed-shape scaling. Superseded exploratory figures are not retained in the publication directories.

Regenerate the publication figures from retained results:

```bash
python analysis/generate_ieee_paper_figures.py
```

## Clean compilation

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

After compilation, inspect every page and submit the PDF to IEEE PDF eXpress before uploading it to CMT.
