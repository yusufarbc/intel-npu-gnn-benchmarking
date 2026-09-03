# Results - Benchmark Data and Figures

Retained benchmark measurements and publication figures.

## Structure

```text
results/
|-- {model_name}/
|   |-- scalability_matrix.csv   # Raw latency data across configurations
|   |-- latency_summary.csv      # Aggregated latency by dataset/device/precision
|   |-- precision_gain.csv       # INT8/FP32 speedup ratios, where available
|   |-- cpu_fallback.json        # Retained device-assignment diagnostics
|   |-- energy.json              # Package-power records, where available
|   |-- run_00/                  # Repeat 1 profiling data
|   |-- run_01/                  # Repeat 2 profiling data
|   `-- run_02/                  # Repeat 3 profiling data
|-- figures/
|   |-- master_results.csv
|   |-- unified_summary.csv
|   |-- comparison_table.csv
|   |-- fig1_latency_comparison.*
|   |-- fig2_int8_speedup_heatmap.*
|   |-- fig3_operator_breakdown.*
|   |-- fig5a_opt_speedup.*
|   `-- fig6_scaling.*
`-- scaling_sweep/
    `-- scaling_sweep.csv         # Retained node/edge scaling measurements
```

## Retention policy

- Numeric CSV/JSON measurements and compact profiling traces are versioned as experimental evidence.
- Generated ONNX binaries, external weight data, per-run preview charts, logs, and sweep run directories are excluded because they can be regenerated and are not needed to evaluate the reported measurements.
- The five figure families listed above support the camera-ready manuscript. Superseded fallback, roofline, density-correlation, and general latency-heatmap outputs are not retained.
- OpenVINO records the integrated graphics device as `GPU`; paper-facing text calls it **iGPU**.
- To regenerate the camera-ready figures from retained results, run `python analysis/generate_ieee_paper_figures.py`.
