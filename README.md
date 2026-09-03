# Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis

**A reproducible characterization of GNN, CNN, and transformer inference on a power-constrained Intel Meteor Lake client platform.**

[![IEEE HPEC 2026](https://img.shields.io/badge/IEEE%20HPEC-2026-00629B.svg)](https://ieee-hpec.org/)
[![Interactive poster](https://img.shields.io/badge/Interactive%20poster-GitHub%20Pages-7B2CBF.svg)](https://yusufarbc.github.io/intel-npu-gnn-benchmarking/)
[![Paper](https://img.shields.io/badge/Paper-camera--ready-B31B1B.svg)](paper/paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EA44F.svg)](LICENSE)

## The study in 30 seconds

| Challenge | Approach | Experiment | Main outcome |
|---|---|---|---|
| Client NPUs are designed around regular tensor execution, while GNNs combine sparse matrix operations, Gather/Scatter, and indirect memory access. | Export the same workloads to ONNX and measure end-to-end inference across the CPU, Arc iGPU, and AI Boost NPU through OpenVINO. | **14 models** × fixed-shape inputs derived from **3 OGB datasets** × **3 backends**, with FP32/INT8 variants, runtime traces, and selected package-power measurements. | The **NPU provides its largest CPU-relative gains on selected dense FP32 models**; the **iGPU generally leads the evaluated GNNs**. INT8 must be validated per model. |

> **Takeaway:** accelerator availability is not the same as verified accelerator execution. A requested NPU configuration may regress, fail to produce an executable artifact, or show an execution anomaly that requires a retained per-operator trace to interpret.

### Key measurements

<p align="center">
  <img src="results/figures/fig1_latency_comparison.svg" width="100%" alt="FP32 latency comparison across the CPU, iGPU, and NPU" />
</p>

<p align="center"><em>FP32 latency across the CPU, integrated GPU (iGPU), and NPU. Lower is better.</em></p>

<table width="100%">
  <thead>
    <tr>
      <th> Dense FP32 highlights</th>
      <th> Sparse GNN and INT8 highlights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <strong>MobileNetV2:</strong> 1.90 ms on NPU, <strong>4.5x</strong> vs. CPU<br>
        <strong>ResNet50:</strong> 3.94 ms on NPU, <strong>8.0x</strong> vs. CPU<br>
        <strong>ViT-Tiny:</strong> 9.10 ms on NPU, <strong>11.4x</strong> vs. CPU
      </td>
      <td>
        <strong>GraphTransformer:</strong> 6.03 ms iGPU vs. 10.72 ms NPU<br>
        <strong>SGC INT8 on NPU:</strong> 173.90 ms, <strong>2.2x slower</strong> than FP32<br>
        <strong>GAT/GATv2 INT8:</strong> NPU compilation failed
      </td>
    </tr>
  </tbody>
</table>

### Deployment decision guide

| Workload or objective | Recommended starting point | Required verification |
|---|---|---|
| Supported, regular FP32 vision model | **NPU** | Confirm native NPU assignment and measure latency |
| Sparse GNN represented by this benchmark | **iGPU** | Compare against CPU and NPU on the target graph |
| INT8 deployment | **No universal winner** | Measure FP32 and INT8; inspect compilation and provider traces |
| Energy-sensitive inference | **Model-specific evaluation** | Measure energy per inference; do not infer isolated NPU power from package telemetry |

## Explore the project

| Resource | What it provides |
|---|---|
| [Interactive poster](https://yusufarbc.github.io/intel-npu-gnn-benchmarking/) | Six-slide visual summary for the IEEE HPEC 2026 poster session |
| [Camera-ready paper](paper/paper.pdf) | Full methodology, analysis, limitations, and references |
| [Benchmark notebook](npu_gnn_benchmarking.ipynb) | End-to-end experiment and figure pipeline |

## Experimental setup

| Dimension | Evaluated configuration |
|---|---|
| Platform | Intel Core Ultra 5 125H, 16 GB LPDDR5x, Windows 11 |
| Backends | 14-core CPU, Intel Arc iGPU, Intel AI Boost NPU 3720 |
| Workloads | 9 GNNs plus 5 CNN/transformer baselines |
| Graph data | ogbn-arxiv, ogbn-products, ogbn-proteins |
| Protocol | Batch 1; 5 warm-ups; 100 timed iterations; 3 independent runs |
| Paper stack | OpenVINO 2024.1 and ONNX Runtime 1.18 |

Latency includes provider dispatch and required host-device transfers after warm-up. Loading, preprocessing, model conversion, and one-time compilation are excluded.

## Run the benchmark

```bash
git clone https://github.com/yusufarbc/intel-npu-gnn-benchmarking.git
cd intel-npu-gnn-benchmarking
python -m venv .venv
python -m pip install -r requirements.txt
jupyter notebook npu_gnn_benchmarking.ipynb
```

The root [`requirements.txt`](requirements.txt) tracks the maintained environment for **new measurements**. It is newer than the paper's OpenVINO 2024.1 / ONNX Runtime 1.18 stack and is not a bit-for-bit historical lockfile.

> NPU and iGPU measurements require a compatible Intel Core Ultra system and working OpenVINO drivers. OpenVINO's provider identifier `GPU` refers to the integrated Arc **iGPU** in this project.

## Evidence boundaries

- Results characterize one Core Ultra 5 125H system, not every Meteor Lake or later NPU.
- The suite covers established reference GNNs, not large, temporal, heterogeneous, or recommendation graphs.
- GNN inference uses fixed 2,708-node, 10,000-edge tensors derived from the OGB source graphs; results are not full-graph OGB measurements.
- Dense baselines use the same synthetic input under each dataset label; those labels are repetitions, not independent dense-model datasets.
- NPU power could not be isolated; package telemetry must not be presented as an NPU-only measurement.
- Latency-derived energy values are heuristic products, not direct active-inference energy measurements.
- FP32 and INT8 are the evaluated precisions; quantized-model accuracy is outside this study.
- Requested-device labels and provider lists do not by themselves prove per-operator assignment.

## Citation

If you use the benchmark suite or results, cite the paper and repository metadata in [`CITATION.cff`](CITATION.cff):

```bibtex
@inproceedings{arabaci2026gnnnpu,
  title     = {Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis},
  author    = {Arabaci, Yusuf Talha and Demiral, Emrullah and Acar, Omer Faruk},
  booktitle = {2026 IEEE High Performance Extreme Computing Conference (HPEC)},
  year      = {2026}
}
```

## Authors and license

**Yusuf Talha Arabacı · Emrullah Demiral · Ömer Faruk Acar**<br>
Department of Software Engineering, Karabük University

Released under the [MIT License](LICENSE). Questions and reproducibility reports are welcome through [GitHub Issues](https://github.com/yusufarbc/intel-npu-gnn-benchmarking/issues).
