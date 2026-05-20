# BENCHMARKING GNN INFERENCE BOTTLENECKS ON INTEL CORE ULTRA NPUS

A comprehensive benchmarking framework for evaluating graph neural network (gnn) inference on intel core ultra (meteor lake) npus, comparing against cpu and integrated gpu (igpu) backends under openvino.

## KEY FEATURES

- **14 Models:** 9 gnns (gcn, gat, gatv2, gin, graphsage, sgc, appnp, graphtransformer, mpnn) + 5 dense baselines (resnet50, mobilenetv2, efficientnet-b0, vit-tiny, bert-tiny)
- **3 Hardware Backends:** Cpu, integrated gpu (xe-lpg), and npu (intel ai boost)
- **3 Real-World Datasets:** Ogbn-arxiv, ogbn-proteins, ogbn-products from the Open Graph Benchmark
- **Int8 Quantization Analysis:** Fp32 vs int8 speedup across all models and devices
- **Operator Profiling:** Per-operator cpu fallback detection and onnx operator composition analysis
- **Publication-Ready Figures:** 7 ieee-format figures (png + svg, 300 dpi)
- **Accompanying Paper:** Latex source for ieee conference submission

## PROJECT STRUCTURE

| Directory | Description |
|-----------|-------------|
| `analysis/` | Python scripts for benchmarking, profiling, and analysis |
| `data/` | Ogb graph datasets (excluded from git) |
| `docs/` | Technical documentation and methodology guides |
| `models/` | Pre-exported onnx models (fp32 and int8) |
| `paper/` | Latex paper source and compiled pdf |
| `results/` | Benchmark outputs, csv matrices, and figures |

## INSTALLATION

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## USAGE

The main entry point is the jupyter notebook:

```bash
jupyter notebook npu_gnn_benchmarking.ipynb
```

Run cells sequentially:
1. **Stage 0:** Global setup (imports, config, helper functions)
2. **Stage 1-A/B/C:** Dataset metadata loading, model inventory, hardware health check
3. **Stage 2:** Unified batch benchmark (all models x datasets x devices)
4. **Stage 3-A/B/C:** Merge results, comparison tables, summary statistics
5. **Stage 4:** 7 publication figures

Standalone script usage:

```bash
python analysis/scalability_analyzer.py --model GCN --device NPU --iterations 100
python analysis/model_prep.py
python analysis/density_sweep.py --devices CPU,GPU,NPU
```

## KEY FINDINGS

- The npu delivers strong fp32 performance for dense vision models (mobilenetv2: 1.97ms, resnet50: 3.92ms)
- Gnn workloads show limited npu advantage (within 15% of cpu latency)
- Int8 quantization degrades npu latency for most models (sgc int8 is 2x slower than fp32)
- Attention-based gnns (gat, gatv2) fail int8 compilation entirely
- Graph density shows no correlation with npu latency (static-shape compilation)

## REPRODUCIBILITY

All benchmark artifacts (scalability matrices, latency summaries, profiling traces, figures) are generated automatically by the notebook. The paper's results can be reproduced by running all cells in sequence.

## LICENSE

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Developed by Yusuf Talha ARABACI (Karabük University)*
