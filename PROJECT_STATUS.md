# Project Status Report

Date: 2026-04-12

## Scope Completed

1. Benchmark core
- Implemented OOP benchmark runner in scripts/benchmark_npu.py.
- Added two graph optimization modes:
  - baseline: ORT_DISABLE_ALL
  - optimized: ORT_ENABLE_EXTENDED
- Added provider fallback ordering for NPU-oriented execution.
- Added latency summary CSV and baseline-vs-optimized chart outputs.

2. Profiling parser and operator-level evidence
- Added scripts/parse_operator_breakdown.py.
- Added scripts/parse_profiling.py as the requested naming-compatible entry point.
- Added operator-level outputs:
  - operator_breakdown_by_mode.csv
  - operator_breakdown.csv
  - operator_breakdown_topk.png
  - operator_top5_speedup.csv
  - disappeared_operators.csv
  - operator_count_delta.csv

3. Scalability and roofline workflow
- Added scripts/run_scalability_study.py.
- Supports multi-model experiments, repeated runs, and statistical summaries.
- Adds graph-size and complexity evidence:
  - node count reduction (|V|)
  - c-factor ratio (optimized/baseline latency)
  - speedup and latency reduction
  - 95% confidence interval estimates
- Adds roofline-style interpretation fields:
  - arithmetic intensity estimate (FLOP/Byte)
  - ridge point
  - memory-bound vs compute-bound label
  - attained GFLOP/s estimate

4. End-to-end automation pipeline
- Added scripts/run_all.py.
- One-command flow now executes:
  - benchmark_npu.py (single-model baseline/optimized)
  - parse_profiling.py (operator-level evidence)
  - run_scalability_study.py (multi-model summary)
  - automatic PNG copy from results/ to paper/figures/

5. Academic paper assets
- Added paper/main.tex.
- Added paper/references.bib.
- Added paper/figures/ directory for result figures.

6. Documentation
- Updated README.md with run commands and expected artifacts.

## Current Output Contract (What You Will Deliver)

When models are present under models/, running the scripts produces:

1. Single-model benchmark artifacts
- results/performance_summary.csv
- results/performance_comparison.png
- results/baseline_profiling.json
- results/optimized_profiling.json

2. Operator analysis artifacts
- results/operator_breakdown_by_mode.csv
- results/operator_breakdown.csv
- results/operator_breakdown_topk.png
- results/operator_top5_speedup.csv
- results/disappeared_operators.csv
- results/operator_count_delta.csv

3. Multi-model scalability artifacts
- results/scalability_matrix.csv
- results/scalability_speedup.png
- results/scalability_latency.png
- per-model per-run folders in results/<model_name>/run_##/

## Remaining Manual Steps

1. Install dependencies in your active environment
- pip install -r requirements.txt

2. Place ONNX models in models/
- Example set: ResNet18, ResNet50, ResNet101 (or any comparable small/medium/large models).

3. Run full pipeline
- python scripts/run_all.py --repeats 3 --iterations 100 --peak-compute-gflops <HW_PEAK> --peak-bandwidth-gbps <HW_BW>

4. Update paper/main.tex result numbers from generated CSV files.

## OpenVINO EP (Windows) Note

If OpenVINO EP is listed but silently falls back to CPU with an error like:

- Error loading `onnxruntime_providers_openvino.dll` (Error 127: procedure not found)

This is often caused by an unrelated `openvino.dll` appearing earlier on your system PATH (e.g. OEM driver folders).

Current fix in scripts/benchmark_npu.py:

- Prepends the Python package OpenVINO runtime folder (`openvino\\libs`) to PATH for the current process
- Adds the same folder via `os.add_dll_directory(...)` and keeps the cookie alive

Optional:

- Set `ORT_OPENVINO_DEVICE` to force device selection (default is `NPU`).

## Notes on Interpretation

- Asymptotic class typically remains near O(N+E) in DAG execution terms.
- Fusion impact is expected in hidden constants via reduced memory traffic and kernel-launch overhead.
- The strongest evidence comes from combining:
  - latency reduction,
  - operator disappearance/merging,
  - node-count reduction,
  - and roofline shift toward higher arithmetic intensity.
