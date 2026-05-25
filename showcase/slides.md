---
theme: default
background: https://cover.sli.dev
info: |
  ## Benchmarking GNN Inference Bottlenecks on Intel Core Ultra NPUs
  Academic presentation for Yusuf Talha Arabacı, Karabük University.
class: text-center
highlighter: shiki
drawings:
  persist: false
transition: slide-left
title: Benchmarking GNN Inference Bottlenecks on Intel Core Ultra NPUs
---
layout: cover
background: /npu_diagram.png
---

# Benchmarking GNN Inference Bottlenecks on Intel Core Ultra NPUs
### Heterogeneous Edge AI Analysis on Meteor Lake SoC

<div class="mt-20 text-center bg-white/80 backdrop-blur-md p-6 rounded-lg border border-slate-200 shadow-lg max-w-2xl mx-auto">
  <div class="font-bold text-lg text-slate-900">
    Yusuf Talha ARABACI, Emrullah DEMİRAL, Ömer Faruk ACAR
  </div>
  <div class="text-slate-800 text-sm mt-1 font-medium">
    Department of Software Engineering, Karabük University, Karabük, Turkey
  </div>
  <div class="text-slate-600 text-xs mt-1">
    yusuftalhaarabaci@hotmail.com | emrullahdemiral@karabuk.edu.tr | farukacar@karabuk.edu.tr
  </div>
</div>

<Glossary :terms="['npu', 'igpu', 'meteor-lake', 'soc']" />

---
layout: default
---

## The Paradox of Edge AI Acceleration
### Dense Optimization vs. Sparse Reality

<div class="grid-cols-2 mt-4">
  <div class="glass-panel">
    <h3 class="text-blue font-semibold">1. Hardware Hype (Dense)</h3>
    <ul>
      <li>Modern SoCs (Intel Meteor Lake, Apple M-series, Qualcomm Snapdragon) integrate Neural Processing Units (NPUs).</li>
      <li>NPUs excel at <strong>dense, regular computations</strong> (e.g., 2D grid convolutions in CNNs, matrix multiplies in Transformers).</li>
      <li>They rely on high <strong>spatial locality</strong> and predictable data streaming to maximize on-chip SRAM reuse.</li>
    </ul>
  </div>

  <div class="glass-panel highlight-box-warning">
    <h3 class="text-rose font-semibold">2. Graph Reality (Sparse)</h3>
    <ul>
      <li>Graph Neural Networks (GNNs) use <strong>neighborhood aggregation</strong> (SpMM/Gather-Scatter) over irregular matrices.</li>
      <li>Memory accesses are dynamic and sparse, disrupting hardware prefetchers.</li>
      <li><strong>Result:</strong> NPUs stall waiting for DRAM transfers, leaving compute units severely underutilized.</li>
    </ul>
  </div>
</div>

<div class="mt-4">
  ```mermaid
  graph LR
    subgraph CNN / Dense Dataflow
      A[Regular 2D Grid] --> B[Spatial Locality & SRAM Reuse]
      B --> C[Compute Bound: High MAC Utilization]
    end
    subgraph GNN / Sparse Dataflow
      D[Irregular Graph] --> E[Random Pointer-Chasing/Gather]
      E --> F[Memory Bound: DRAM Latency Wall]
    end
    style C fill:#dcfce7,stroke:#16a34a,stroke-width:1px
    style F fill:#ffe4e6,stroke:#be123c,stroke-width:1px
  ```
</div>

<Glossary :terms="['gnn', 'compute-bound', 'memory-bound', 'stalls', 'locality']" />

---
layout: default
---

## Key Technical Inquiries
### Core Research Questions of the Study

<div class="mt-8 flex flex-col gap-4">
  <div class="highlight-box highlight-box-info">
    <div class="highlight-box-title">RQ 1: On-Device Efficiency & Parity</div>
    <div class="text-sm">
      How efficient are consumer-grade NPUs for sparse GNN workloads compared to traditional general-purpose CPUs and integrated GPUs (iGPUs) on client laptops?
    </div>
  </div>

  <div class="highlight-box highlight-box-warning">
    <div class="highlight-box-title">RQ 2: The Quantization Speedup Fallacy</div>
    <div class="text-sm">
      Does 8-bit quantization (INT8) consistently deliver the advertised 4x acceleration on NPUs for graph models, or does it trigger performance and compilation regressions?
    </div>
  </div>

  <div class="highlight-box highlight-box-success">
    <div class="highlight-box-title">RQ 3: Compiler Maturity & Operator Fusion limits</div>
    <div class="text-sm">
      How robust is the OpenVINO compiler toolchain when lowering graph propagation primitives (e.g., dynamic Gather/Scatter, index operations) onto NPU microarchitectures?
    </div>
  </div>
</div>

<Glossary :terms="['openvino', 'operator-fusion', 'cpu-fallback']" />

---
layout: default
---

## Hardware & Software Methodology
### Experimental Configuration and Execution Protocol

<div class="grid-cols-2 mt-6">
  <div class="glass-panel">
    <h3 class="font-semibold text-blue">Hardware Infrastructure</h3>
    <ul>
      <li><strong>Processor:</strong> Intel Core Ultra 5 125H (Meteor Lake)</li>
      <li><strong>Heterogeneous Backends:</strong>
        <ul>
          <li><span class="stat-badge badge-cpu">CPU</span> 14 Cores (4P + 8E + 2LPE), AVX-512 vector extensions</li>
          <li><span class="stat-badge badge-gpu">iGPU</span> Intel Arc Graphics (7 Xe-cores, Xe-LPG)</li>
          <li><span class="stat-badge badge-npu">NPU</span> Intel AI Boost NPU (3720 series tile)</li>
        </ul>
      </li>
      <li><strong>System Memory:</strong> 16 GB LPDDR5x RAM</li>
    </ul>
  </div>

  <div class="glass-panel">
    <h3 class="font-semibold text-blue">Software & Measurement Protocol</h3>
    <ul>
      <li><strong>Compiler Toolchain:</strong> OpenVINO 2024.1 with native NPU plugin.</li>
      <li><strong>Execution Provider:</strong> ONNX Runtime 1.18.</li>
      <li><strong>Measurement Loop:</strong>
        <ul>
          <li>5 warm-up iterations.</li>
          <li>100 timed iterations.</li>
          <li>3 independent repeats (averages and standard deviations).</li>
        </ul>
      </li>
      <li><strong>Telemetry:</strong> Intel SoCWatch PMT CLI (Windows Virtualization-Based Security bypass) for package power.</li>
    </ul>
  </div>
</div>

<Glossary :terms="['socwatch', 'warm-up']" />

---
layout: default
---

## Evaluated Models and OGB Datasets
### Workload Characterization

<div class="grid-cols-2 mt-2">
  <div class="glass-panel">
    <h3 class="font-semibold text-blue">14 Model Architectures</h3>
    <div class="text-xs">
      <strong>Graph Neural Networks (9):</strong>
      <ul>
        <li>Spectral GNNs: GCN, SGC</li>
        <li>Spatial/Expressive GNNs: GIN, GraphSAGE, MPNN</li>
        <li>Propagation/Attentional: APPNP, GAT, GATv2, GraphTransformer</li>
      </ul>
      <strong class="mt-2 block">Dense Baselines (5):</strong>
      <ul>
        <li>Convolutions: ResNet50, MobileNetV2, EfficientNet-B0</li>
        <li>Attention: ViT-Tiny, BERT-Tiny</li>
      </ul>
    </div>
  </div>

  <div class="glass-panel">
    <h3 class="font-semibold text-blue">3 Real-world Graph Datasets</h3>
    <div class="text-xs mb-2">
      Drawn from the Open Graph Benchmark (<a href="file:///c:/Users/yusuf/Github/npu-graph-opt-benchmarking/results/figures/dataset_stats.csv">dataset_stats.csv</a>):
    </div>
    <table>
      <thead>
        <tr>
          <th>Dataset</th>
          <th>Nodes</th>
          <th>Edges</th>
          <th>Features</th>
          <th>Edges/Node</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>ogbn-arxiv</strong></td>
          <td>169K</td>
          <td>1.16M</td>
          <td>128</td>
          <td>6.9</td>
        </tr>
        <tr>
          <td><strong>ogbn-products</strong></td>
          <td>2.45M</td>
          <td>61.9M</td>
          <td>100</td>
          <td>25.3</td>
        </tr>
        <tr>
          <td><strong>ogbn-proteins</strong></td>
          <td>87.6K</td>
          <td>39.6M</td>
          <td>8</td>
          <td>451.7</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<Glossary :terms="['sparse-graph', 'dense-graph', 'adjacency-matrix']" />

---
layout: default
---

## Inference Latency: NPU vs. CPU vs. iGPU
### High-Contrast Performance Profile (FP32)

<div class="grid-cols-2 mt-4">
  <div>
    <ul>
      <li><strong>Dense Acceleration:</strong> The NPU achieves massive speedups for convolutional and dense attention models over CPU:
        <ul>
          <li>MobileNetV2: <strong>1.90 ms</strong> on NPU vs. 8.60 ms on CPU (<strong>4.5&times;</strong>)</li>
          <li>ResNet50: <strong>3.94 ms</strong> on NPU vs. 31.47 ms on CPU (<strong>8.0&times;</strong>)</li>
          <li>ViT-Tiny: <strong>9.10 ms</strong> on NPU vs. 104.13 ms on CPU (<strong>11.4&times;</strong>)</li>
        </ul>
      </li>
      <li class="mt-4"><strong>GNN Execution Parity:</strong> GNN models show close CPU-NPU parity (within &plusmn;6%), because they are DRAM-bound.</li>
      <li class="mt-4"><strong>iGPU Dominance for GNNs:</strong> The integrated GPU is the overall winner for GNN workloads (GraphTransformer: <strong>6.03 ms</strong> on GPU vs. 10.72 ms on NPU).</li>
    </ul>
  </div>
  
  <div class="flex flex-col justify-center items-center">
    <img src="/figures/fig1_latency_comparison.png" class="slide-img" alt="FP32 Latency Comparison" />
    <span class="text-xs text-slate-500 mt-2">Figure 1: Latency comparison across compute backends (<a href="file:///c:/Users/yusuf/Github/npu-graph-opt-benchmarking/results/figures/comparison_table.csv">comparison_table.csv</a>)</span>
  </div>
</div>

<Glossary :terms="['npu', 'igpu', 'memory-bound']" />

---
layout: default
---

## Counter-Intuitive INT8 Quantization Results
### Quantization Paradox & Silent Fallbacks

<div class="grid-cols-2 mt-4">
  <div>
    <ul>
      <li><strong>Minor to Negative Gains on NPU:</strong>
        <ul>
          <li>GCN (1.04&times;), GraphSAGE (1.05&times;) show marginal speedups.</li>
          <li>SGC INT8 displays a massive <strong>2.2&times; performance regression</strong> over FP32 (173.90 ms vs. 78.59 ms).</li>
        </ul>
      </li>
      <li class="mt-2"><strong>Compilation Failures:</strong>
        <ul>
          <li>GAT, GATv2, and EfficientNet-B0 fail compiler lowering completely due to unsupported operators.</li>
        </ul>
      </li>
      <li class="mt-2"><strong>Silent CPU Fallbacks:</strong>
        <ul>
          <li>Quantized vision models (MobileNetV2, BERT-Tiny) silently fallback to CPU execution, yielding misleading latencies.</li>
        </ul>
      </li>
    </ul>
    <div class="highlight-box highlight-box-warning text-xs mt-2">
      <strong>Quantization Rule:</strong> Quantization decreases arithmetic bitwidth but does not solve irregular memory aggregation stalls.
    </div>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="/figures/fig2_int8_speedup_heatmap.png" class="slide-img" alt="INT8 Speedup Heatmap" />
    <span class="text-xs text-slate-500 mt-2">Figure 2: INT8 speedup heatmap. Red (< 1.0) represents performance degradation.</span>
  </div>
</div>

<Glossary :terms="['int8', 'regression', 'cpu-fallback']" />

---
layout: default
---

## Structured vs. Irregular Attention Patterns
### ViT-Tiny vs. GraphTransformer on NPU

<div class="grid-cols-2 mt-6">
  <div class="glass-panel">
    <h3 class="font-semibold text-emerald">Structured Attention (ViT-Tiny)</h3>
    <ul>
      <li>Computes self-attention over a fixed 2D grid of patches.</li>
      <li>Highly structured, static, and predictable memory strides.</li>
      <li>Enables maximum kernel fusion and compiler optimization.</li>
      <li><strong>Speedup over CPU:</strong> <span class="text-emerald font-bold">11.4&times; (9.10 ms vs. 104.13 ms)</span></li>
    </ul>
  </div>

  <div class="glass-panel highlight-box-warning">
    <h3 class="font-semibold text-rose">Irregular Attention (GraphTransformer)</h3>
    <ul>
      <li>Computes attention over dynamic graph neighborhoods.</li>
      <li>Irregular indices and sizes lead to random DRAM queries.</li>
      <li>Static shape compilation prevents data-dependent optimization.</li>
      <li><strong>Speedup over CPU:</strong> <span class="text-rose font-bold">1.0&times; (10.72 ms vs. 10.69 ms)</span></li>
    </ul>
  </div>
</div>

<div class="glass-panel mt-4 text-center text-xs">
  Even with <strong>30&times; fewer parameters</strong> (0.18M vs. 5.7M), GraphTransformer fails to gain any speedup on the NPU compared to ViT-Tiny.
</div>

<Glossary :terms="['structured-attention', 'irregular-attention']" />

---
layout: default
---

## Power Consumption and Throughput per Watt
### SoCWatch Package-Level Telemetry

<div class="grid-cols-2 mt-4">
  <div>
    <ul>
      <li><strong>Package Power Profiles:</strong>
        <ul>
          <li>iGPU draws slightly higher peak package power (+7.3% for GCN) but executes faster than CPU, yielding equal energy-per-inference.</li>
        </ul>
      </li>
      <li class="mt-4"><strong>Energy per Inference ($\text{mJ}$):</strong>
        <ul>
          <li>GCN on CPU benefits from INT8: 16.8% power drop, <strong>18.4% energy reduction</strong>.</li>
          <li>MPNN on CPU exhibits regression: INT8 increases latency (+63%), leading to <strong>59% higher energy</strong> (301.1 mJ vs. 189.3 mJ).</li>
        </ul>
      </li>
    </ul>
  </div>

  <div>
    <div class="text-xs mb-2"><strong>Package Power and Energy Results:</strong></div>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Config</th>
          <th>Avg Power (mW)</th>
          <th>Latency (ms)</th>
          <th>Energy (mJ)</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background-color: #f8fafc">
          <td><strong>GCN</strong></td>
          <td>CPU FP32</td>
          <td>11,629</td>
          <td>7.73</td>
          <td>89.9</td>
        </tr>
        <tr>
          <td><strong>GCN</strong></td>
          <td>CPU INT8</td>
          <td>9,675</td>
          <td>7.59</td>
          <td><strong>73.4 (-18.4%)</strong></td>
        </tr>
        <tr style="background-color: #f8fafc">
          <td><strong>GCN</strong></td>
          <td>GPU FP32</td>
          <td>12,482</td>
          <td>6.94</td>
          <td>86.6</td>
        </tr>
        <tr>
          <td><strong>MPNN</strong></td>
          <td>CPU FP32</td>
          <td>9,357</td>
          <td>20.23</td>
          <td>189.3</td>
        </tr>
        <tr style="background-color: #f8fafc">
          <td><strong>MPNN</strong></td>
          <td>CPU INT8</td>
          <td>9,139</td>
          <td>32.94</td>
          <td><strong>301.1 (+59.0%)</strong></td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<Glossary :terms="['socwatch', 'throughput-watt']" />

---
layout: default
---

## Impact of Graph Density on NPU Efficiency
### Density-Latency Decoupling

<div class="grid-cols-2 mt-4">
  <div>
    <ul>
      <li><strong>OGB Dataset Sweep:</strong>
        <ul>
          <li>Tests graphs spanning two orders of magnitude in average node degree: 6.9 (arxiv), 25.3 (products), 451.7 (proteins).</li>
        </ul>
      </li>
      <li class="mt-4"><strong>Static-Shape Flatline ($r \approx 0$):</strong>
        <ul>
          <li>Surprisingly, latency remains flat regardless of graph density.</li>
          <li>ONNX Runtime compiles models with static tensor dimensions.</li>
          <li>The NPU executes the fixed-size compilation layout at constant throughput, missing any potential sparsity-related computational shortcuts.</li>
        </ul>
      </li>
    </ul>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="/figures/fig7_density_vs_latency.png" class="slide-img" alt="NPU Latency vs Graph Density" />
    <span class="text-xs text-slate-500 mt-2">Figure 3: Flat latency curves across density levels (<a href="file:///c:/Users/yusuf/Github/npu-graph-opt-benchmarking/results/figures/fig7_density_vs_latency.svg">fig7_density_vs_latency.svg</a>)</span>
  </div>
</div>

<Glossary :terms="['sparse-graph', 'adjacency-matrix', 'spmm']" />

---
layout: default
---

## Why Do GNNs Struggle on Consumer NPUs?
### Key Architectural Bottlenecks

<div class="grid-cols-2 mt-4">
  <div>
    <ul>
      <li><strong>1. Memory Wall (DRAM Bandwidth):</strong>
        <ul>
          <li>GNNs belong to the memory-bound region (0.1–10 FLOP/byte).</li>
          <li>Reducing precision to INT8 decreases arithmetic workload, but fails to mitigate pointer-chasing and gather latency.</li>
        </ul>
      </li>
      <li class="mt-2"><strong>2. Compiler Fusion Limits:</strong>
        <ul>
          <li>Indirect indexing operations (Gather, Scatter) prevent compile-time static flow optimization.</li>
          <li>The OpenVINO NPU compiler plugin applies fewer fusion passes to GNN subgraphs.</li>
        </ul>
      </li>
      <li class="mt-2"><strong>3. Operator Coverage & Fallbacks:</strong>
        <ul>
          <li>MPNN fails to execute on NPU due to lack of support for <code>index_add_</code> operators, forcing 100% CPU fallback.</li>
        </ul>
      </li>
    </ul>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="/figures/fig4_cpu_fallback_heatmap.png" class="slide-img" alt="CPU Fallback Heatmap" />
    <span class="text-xs text-slate-500 mt-2">Figure 4: Operator-level CPU fallback fraction per model.</span>
  </div>
</div>

<Glossary :terms="['memory-bound', 'operator-fusion', 'cpu-fallback', 'spmm']" />

---
layout: default
---

## Practical Guidance for Edge AI Deployments
### System Guidelines and Future Roadmap

<div class="mt-6 flex flex-col gap-4">
  <div class="highlight-box highlight-box-success">
    <div class="highlight-box-title">✔ Recommended: Dense Vision on NPU (INT8)</div>
    <div class="text-sm">
      Deploy traditional vision networks (ResNet50, MobileNetV2) and regular transformers to the <strong>NPU</strong> using <strong>INT8 quantization</strong>. The compile pipeline is mature and provides extreme energy efficiency.
    </div>
  </div>

  <div class="highlight-box highlight-box-warning">
    <div class="highlight-box-title">⚠ Caution: GNN Workloads on GPU (FP32)</div>
    <div class="text-sm">
      Deploy GNNs (GCN, GraphSAGE, GraphTransformer) to the <strong>iGPU</strong> in <strong>FP32</strong>. Avoid NPU execution due to compilation failures, memory bottlenecks, and silent fallbacks under current toolchains (OpenVINO 2024.1).
    </div>
  </div>
</div>

<div class="mt-8 text-center text-slate-700 font-semibold">
  Thank you! Questions & Answers.
  <div class="text-xs text-slate-500 mt-2">
    Karabük University Yazılım Mühendisliği Bölümü
  </div>
</div>

<Glossary :terms="['npu', 'igpu', 'int8', 'fp32']" />
