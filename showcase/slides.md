---
theme: default
layout: default
info: |
  ## Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis
  Academic presentation for Yusuf Talha Arabacı, Karabük University.
class: text-left
highlighter: shiki
drawings:
  persist: false
transition: fade
routerMode: hash
title: "Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis"
---

<div class="flex flex-col items-center justify-center h-full gap-3">
  <div class="grid grid-cols-12 gap-6 w-full items-center">
    <div class="col-span-7 flex flex-col justify-center">
      <h1 class="text-3xl font-extrabold text-blue-800 leading-tight">
        Benchmarking GNN Inference on the Intel Core Ultra NPU
      </h1>
      <h4 class="text-slate-500 font-medium text-sm mt-1">A Latency, Quantization, and Energy Analysis</h4>
      <h3 class="text-slate-600 font-semibold mt-2 text-lg leading-relaxed">
        Heterogeneous Edge AI Analysis on Meteor Lake SoC
      </h3>
    </div>
    <div class="col-span-5 flex justify-center items-center">
      <div class="p-2 bg-white border border-slate-200 rounded-xl shadow-md hover:shadow-lg transition-shadow">
        <img src="./public/meteor-lake-architecture.jpg" class="max-h-70 object-contain rounded" />
        <div class="text-center text-xs text-slate-500 mt-2 font-medium">
          Figure 1: Intel Meteor Lake heterogeneous SoC
        </div>
      </div>
    </div>
  </div>
  <div class="p-5 bg-slate-50 border border-slate-200 rounded-lg shadow-sm w-full max-w-2xl">
    <div class="font-bold text-base text-slate-900 text-center">
      Yusuf Talha ARABACI, Emrullah DEMİRAL, Ömer Faruk ACAR
    </div>
    <div class="text-slate-700 text-xs mt-1 text-center font-medium">
      Department of Software Engineering, Karabük University, Karabük, Turkey
    </div>
    <div class="text-slate-500 text-xs mt-0.5 text-center">
      yusuftalhaarabaci@hotmail.com | emrullahdemiral@karabuk.edu.tr | farukacar@karabuk.edu.tr
    </div>
  </div>
</div>

<Glossary :terms="['npu', 'igpu', 'meteor-lake', 'soc']" />

---
layout: default
class: rq-slide
---

## Agenda
### Presentation Outline

<div class="flex flex-col gap-3 mt-4">
  <div class="glass-panel">
    <div class="flex items-center gap-3">
      <span class="text-lg font-bold text-blue" style="min-width:2rem">1</span>
      <div><strong>Background &amp; Motivation</strong> — NPU promise, GNN challenges, study goals</div>
    </div>
  </div>
  <div class="glass-panel">
    <div class="flex items-center gap-3">
      <span class="text-lg font-bold text-blue" style="min-width:2rem">2</span>
      <div><strong>Methodology</strong> — Hardware platform, 14 models, 3 OGB datasets, measurement protocol</div>
    </div>
  </div>
  <div class="glass-panel">
    <div class="flex items-center gap-3">
      <span class="text-lg font-bold text-blue" style="min-width:2rem">3</span>
      <div><strong>Results</strong> — Latency, INT8 quantization, operator analysis, roofline, density, energy</div>
    </div>
  </div>
  <div class="glass-panel">
    <div class="flex items-center gap-3">
      <span class="text-lg font-bold text-rose" style="min-width:2rem">4</span>
      <div><strong>Key Findings &amp; Bottlenecks</strong> — Why INT8 fails, CPU fallback, comparative landscape</div>
    </div>
  </div>
  <div class="glass-panel">
    <div class="flex items-center gap-3">
      <span class="text-lg font-bold text-emerald" style="min-width:2rem">5</span>
      <div><strong>Practical Guidance</strong> — Deployment recommendations, future work, limitations</div>
    </div>
  </div>
</div>

<Glossary :terms="['npu', 'gnn', 'openvino']" />

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
      <li>GNN neighborhood aggregation aggregates node features irregularly:</li>
    </ul>
$$h_v^{(l+1)} = \text{UPDATE}^{(l)} \left( h_v^{(l)}, \text{AGGREGATE}^{(l)} \left( \{ h_u^{(l)} : u \in \mathcal{N}(v) \} \right) \right)$$
    <ul>
      <li>Computations rely on irregular Sparse-Dense Matrix Multiplications (SpMM):</li>
    </ul>
$$Y = A \cdot X$$
    <ul>
      <li>Memory accesses are dynamic and sparse, disrupting hardware prefetchers.</li>
      <li><strong>Result:</strong> NPUs stall waiting for DRAM transfers, leaving compute units underutilized.</li>
    </ul>
  </div>
</div>

<Glossary :terms="['gnn', 'compute-bound', 'memory-bound', 'stalls', 'locality', 'scatter-gather']" />

---
layout: default
---

## Key Technical Inquiries
### Core Research Questions

<div class="mt-2 flex flex-col gap-2">
  <div class="highlight-box highlight-box-info">
    <div class="highlight-box-title">RQ 1: On-Device Efficiency & Parity</div>
    <div class="text-sm">How efficient are consumer NPUs for sparse GNN workloads vs. CPU and iGPU on laptops?</div>
  </div>

  <div class="highlight-box highlight-box-warning">
    <div class="highlight-box-title">RQ 2: The Quantization Speedup Fallacy</div>
    <div class="text-sm">Does INT8 deliver advertised 4× acceleration on NPUs, or does it trigger regressions?</div>
  </div>

  <div class="highlight-box highlight-box-success">
    <div class="highlight-box-title">RQ 3: Compiler Maturity & Operator Fusion Limits</div>
    <div class="text-sm">How robust is OpenVINO when lowering Gather/Scatter onto NPU microarchitectures?</div>
  </div>
</div>

<Glossary :terms="['openvino', 'operator-fusion', 'cpu-fallback', 'scatter-gather']" />

---
layout: default
---

## Dense vs. Sparse Dataflow
### Visual Comparison

```mermaid {scale: 0.9}
graph LR
  subgraph "CNN / Dense Dataflow"
    A[Regular 2D Grid] --> B[Spatial Locality & SRAM Reuse]
    B --> C[Compute Bound: High MAC Utilization]
  end
  subgraph "GNN / Sparse Dataflow"
    D[Irregular Graph] --> E[Random Pointer-Chasing/Gather]
    E --> F[Memory Bound: DRAM Latency Wall]
  end
  style C fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px
  style F fill:#ffe4e6,stroke:#be123c,stroke-width:1.5px
```

<Glossary :terms="['compute-bound', 'memory-bound', 'scatter-gather']" />

---
layout: default
---

## Hardware & Software Methodology
### Experimental Configuration

<div class="grid-cols-2 mt-4">
  <div class="glass-panel">
    <h3 class="font-semibold text-blue">Hardware Platform</h3>
    <ul>
      <li><strong>Processor:</strong> Intel Core Ultra 5 125H (Meteor Lake)</li>
      <li><strong>Backends:</strong>
        <span class="stat-badge badge-cpu">CPU</span> 14 Cores (4P+8E+2LPE)
        <span class="stat-badge badge-gpu">iGPU</span> 7 Xe-cores
        <span class="stat-badge badge-npu">NPU</span> AI Boost 3720
      </li>
      <li><strong>Memory:</strong> 16 GB LPDDR5x</li>
    </ul>
  </div>

  <div class="glass-panel">
    <h3 class="font-semibold text-blue">Software Stack</h3>
    <ul>
      <li><strong>Framework:</strong> OpenVINO 2024.1 + ONNX Runtime 1.18</li>
      <li><strong>Protocol:</strong> 5 warm-up, 100 timed iterations, 3 repeats</li>
      <li><strong>Telemetry:</strong> Intel SoCWatch PMT CLI for package power</li>
    </ul>
  </div>
</div>

<Glossary :terms="['socwatch', 'warm-up', 'lpddr5x', 'xe-lpg']" />

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
      Drawn from the Open Graph Benchmark (dataset_stats.csv):
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

<Glossary :terms="['sparse-graph', 'dense-graph', 'adjacency-matrix', 'ogb']" />

---
layout: default
---

## Inference Latency: NPU vs. CPU vs. iGPU
### High-Contrast Performance Profile (FP32)

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>Dense Acceleration:</strong> NPU massively accelerates vision models:
        <ul>
          <li>MobileNetV2: <strong>1.90 ms</strong> NPU vs. 8.60 ms CPU (<strong>4.5&times;</strong>)</li>
          <li>ResNet50: <strong>3.94 ms</strong> vs. 31.47 ms (<strong>8.0&times;</strong>)</li>
          <li>ViT-Tiny: <strong>9.10 ms</strong> vs. 104.13 ms (<strong>11.4&times;</strong>)</li>
        </ul>
      </li>
      <li><strong>GNN Parity:</strong> Close CPU-NPU (&plusmn;6%) — DRAM-bound.</li>
      <li><strong>iGPU leads GNNs:</strong> GraphTransformer 6.03 ms GPU vs. 10.72 ms NPU.</li>
    </ul>
  </div>
  
  <div class="flex flex-col justify-center items-center">
    <img src="./public/figures/fig1_latency_comparison.svg" class="slide-img" />
    <span class="text-xs text-slate-500 mt-1">Figure 1: Latency across backends</span>
  </div>
</div>

<Glossary :terms="['npu', 'igpu', 'memory-bound', 'vit']" />

---
layout: default
---

## Counter-Intuitive INT8 Quantization Results
### Quantization Paradox & Silent Fallbacks

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>Minor-to-Negative Gains on NPU:</strong> GCN (1.04&times;), GraphSAGE (1.05&times;) marginal; SGC shows <strong>2.2&times; regression</strong> (173.9 vs 78.6 ms).</li>
      <li><strong>Compilation Failures:</strong> GAT, GATv2, EfficientNet-B0 fail INT8 lowering entirely.</li>
      <li><strong>Silent CPU Fallback:</strong> Quantized vision models (MobileNetV2, BERT-Tiny) silently run on CPU, giving misleading "NPU" latencies.</li>
    </ul>
    <div class="highlight-box highlight-box-warning text-xs mt-1">
      <strong>Why?</strong> INT8 does not reduce DRAM traffic for memory-bound sparse workloads. Dynamic quantization patterns collide with the NPU's static-shape architecture.
    </div>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="./public/figures/fig2_int8_speedup_heatmap.svg" class="slide-img" />
    <span class="text-xs text-slate-500 mt-1">Figure 2: INT8 speedup heatmap. Red (&lt; 1.0) = degradation.</span>
  </div>
</div>

<Glossary :terms="['int8', 'regression', 'cpu-fallback', 'static-shape', 'scatter-gather']" />

---
layout: default
---

## Structured vs. Irregular Attention Patterns
### ViT-Tiny vs. GraphTransformer on NPU

<div class="grid-cols-2 mt-2">
  <div class="glass-panel">
    <h3 class="font-semibold text-emerald" style="font-size:0.85rem">Structured (ViT-Tiny)</h3>
    <ul>
      <li>Self-attention over fixed 2D grid patches.</li>
      <li>Static, predictable strides → max fusion.</li>
      <li><strong>NPU speedup:</strong> <span class="text-emerald font-bold">11.4&times; vs CPU</span></li>
    </ul>
  </div>

  <div class="glass-panel highlight-box-warning">
    <h3 class="font-semibold text-rose" style="font-size:0.85rem">Irregular (GraphTransformer)</h3>
    <ul>
      <li>Attention over dynamic graph neighborhoods.</li>
      <li>Irregular indices → random DRAM queries.</li>
      <li><strong>NPU speedup:</strong> <span class="text-rose font-bold">1.0&times; vs CPU</span></li>
    </ul>
  </div>
</div>

<div class="glass-panel text-center text-xs mt-1">
  Despite <strong>30&times; fewer parameters</strong> (0.18M vs 5.7M), GraphTransformer gains no NPU speedup.
</div>

<Glossary :terms="['structured-attention', 'irregular-attention', 'vit']" />

---
layout: default
---

## Roofline Analysis & Computational Efficiency
### Memory Bandwidth Wall vs. Compute Saturation

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>Intensity</strong> <KaTeX math="= \text{FLOPs} / \text{Bytes}" /> — operational density per memory transfer.</li>
      <li><strong>Memory-bound GNNs:</strong> 0.1–10 FLOP/byte — limited by LPDDR5x bandwidth, explaining no NPU gain.</li>
      <li><strong>Compute-bound vision:</strong> CNNs/ViT saturate NPU's MAC arrays in higher intensity zones.</li>
    </ul>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="./public/figures/fig5b_roofline.svg" class="slide-img" />
    <span class="text-xs text-slate-500 mt-1">Figure 5: Throughput vs. arithmetic intensity</span>
  </div>
</div>

<Glossary :terms="['compute-bound', 'memory-bound', 'intensity', 'roofline', 'lpddr5x']" />

---
layout: default
---

## Power Consumption and Throughput per Watt
### SoCWatch Package-Level Telemetry

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>Formula:</strong> <KaTeX math="E_{\text{inf}} = P_{\text{package}} \times t_{\text{latency}}" /></li>
      <li><strong>iGPU:</strong> +7.3% peak power vs CPU, but faster — equal energy/inference.</li>
      <li><strong>INT8 on CPU:</strong> GCN −18.4% energy (73.4 vs 89.9 mJ); MPNN +59% (301.1 vs 189.3 mJ).</li>
    </ul>
  </div>

  <div>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Config</th>
          <th>Power (mW)</th>
          <th>Lat (ms)</th>
          <th>Energy (mJ)</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background-color: #f8fafc">
          <td><strong>GCN</strong></td>
          <td>CPU FP32</td><td>11,629</td><td>7.73</td><td>89.9</td>
        </tr>
        <tr>
          <td><strong>GCN</strong></td>
          <td>CPU INT8</td><td>9,675</td><td>7.59</td><td><strong>73.4</strong></td>
        </tr>
        <tr style="background-color: #f8fafc">
          <td><strong>GCN</strong></td>
          <td>GPU FP32</td><td>12,482</td><td>6.94</td><td>86.6</td>
        </tr>
        <tr>
          <td><strong>MPNN</strong></td>
          <td>CPU FP32</td><td>9,357</td><td>20.23</td><td>189.3</td>
        </tr>
        <tr style="background-color: #f8fafc">
          <td><strong>MPNN</strong></td>
          <td>CPU INT8</td><td>9,139</td><td>32.94</td><td><strong>301.1</strong></td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<Glossary :terms="['socwatch', 'throughput-watt']" />

---
layout: default
---

## Cross-Platform Latency Overview
### Heatmap Across Models, Devices, and Precisions

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>iGPU Dominance:</strong> Lowest latency across GNNs and vision.</li>
      <li><strong>NPU Sweet Spot:</strong> Dense vision (ResNet50, MobileNetV2, ViT) — 4.5–11.4× vs CPU.</li>
      <li><strong>GNN Parity:</strong> All backends cluster tightly — memory-bound bottleneck is platform-independent.</li>
    </ul>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="./public/figures/fig8_latency_heatmap.svg" class="slide-img" />
    <span class="text-xs text-slate-500 mt-1">Figure 8: FP32 latency heatmap (14 models &times; 3 backends)</span>
  </div>
</div>

<Glossary :terms="['npu', 'igpu', 'memory-bound', 'onnx']" />

---
layout: default
---

## Scaling Characteristics on NPU
### Graph Size vs. Latency

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>Flat latency:</strong> Unlike CPU/GPU, NPU latency stays constant across node/edge counts.</li>
      <li><strong>Why?</strong> ONNX Runtime compiles with statically shaped tensors — fixed throughput regardless of input.</li>
      <li><strong>Trade-off:</strong> Predictable latency, but no benefit from sparser subgraphs.</li>
    </ul>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="./public/figures/fig6_scaling.svg" class="slide-img" />
    <span class="text-xs text-slate-500 mt-1">Figure 6: Flat NPU latency across graph sizes</span>
  </div>
</div>

<Glossary :terms="['memory-bound', 'npu', 'static-shape', 'onnx']" />

---
layout: default
---

## Impact of Graph Density on NPU Efficiency
### Density-Latency Decoupling

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>Pearson <KaTeX math="r \approx -0.00" /></strong> — no correlation between density &amp; latency on NPU.</li>
      <li><strong>Dataset sweep:</strong> arxiv (6.9 edges/node), products (25.3), proteins (451.7) — two orders of magnitude.</li>
      <li><strong>Static-shape compilation</strong> forces constant throughput regardless of sparsity.</li>
    </ul>
    <div class="highlight-box highlight-box-warning text-xs mt-1">
      <strong>🔑</strong> NPU latency is <strong>completely decoupled</strong> from graph density. Unlike CPU/GPU, sparser subgraphs yield no speedup.
    </div>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="./public/figures/fig7_density_vs_latency.svg" class="slide-img" />
    <span class="text-xs text-slate-500 mt-1">Figure 7: Flat latency across density levels</span>
  </div>
</div>

<Glossary :terms="['sparse-graph', 'adjacency-matrix', 'spmm', 'static-shape']" />

---
layout: default
---

## Why Do GNNs Struggle on Consumer NPUs?
### Key Architectural Bottlenecks

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>1. Memory Wall:</strong> GNNs are memory-bound (0.1–10 FLOP/byte). 2–4× more shape operators (Gather, Scatter) than vision models — poorly served by NPU's streaming-dataflow.</li>
      <li><strong>2. Compiler Fusion Limits:</strong> Indirect indexing (Gather, Scatter) prevents static flow optimization; fewer fusion passes on GNN subgraphs.</li>
      <li><strong>3. Operator Coverage:</strong> MPNN fails entirely on NPU — <code>index_add_</code> unsupported, forcing 100% CPU fallback.</li>
    </ul>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="./public/figures/fig4_cpu_fallback_heatmap.svg" class="slide-img" />
    <span class="text-xs text-slate-500 mt-1">Figure 4: CPU fallback fraction per model.</span>
  </div>
</div>

<Glossary :terms="['memory-bound', 'operator-fusion', 'cpu-fallback', 'spmm', 'scatter-gather']" />

---
layout: default
---

## Operator Composition Analysis
### GNNs vs. Vision Models

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>GNNs:</strong> MatMul (30–60%), 2–4× more Gather/Scatter/Reshape than vision.</li>
      <li><strong>Vision:</strong> Conv (50–80%), minimal shape manipulation.</li>
      <li><strong>Insight:</strong> Gather/Scatter irregularity clashes with NPU's streaming-dataflow.</li>
    </ul>
  </div>

  <div class="flex flex-col justify-center items-center">
    <img src="./public/figures/fig3_operator_breakdown.svg" class="slide-img" />
    <span class="text-xs text-slate-500 mt-1">Figure 3: Operator composition — GNNs vs. vision</span>
  </div>
</div>

<Glossary :terms="['spmm', 'operator-fusion', 'cpu-fallback', 'scatter-gather']" />

---
layout: default
---

## Intel AI Boost: Failure Mode Analysis
### Silent CPU Fallback as a Toolchain Maturity Signal

<div class="mt-2">
  <div class="glass-panel highlight-box-warning">
    <h3 class="font-semibold text-rose">Observed Failure Modes on Meteor Lake NPU</h3>
    <ul>
      <li><strong>Silent CPU Fallback:</strong> OpenVINO silently dispatches unsupported quantized ops to CPU — MobileNetV2, ResNet50, BERT-Tiny INT8 report misleading "NPU" latencies identical to CPU execution.</li>
      <li><strong>Compilation Rejection:</strong> GAT, GATv2, EfficientNet-B0 fail INT8 lowering entirely — dynamic Gather/Scatter unsupported in NPU quantized operator set.</li>
      <li><strong>Full Backend Bypass:</strong> MPNN's <code>index_add_</code> and <code>scatter_mean</code> operators lie outside the NPU plugin — 100% CPU fallback regardless of designated backend.</li>
    </ul>
  </div>

  <div class="glass-panel mt-2">
    <h3 class="font-semibold text-blue">Root Cause</h3>
    <ul>
      <li>Quantization toolchain assumes dense, regular computation graphs — GNNs violate this contract at every level.</li>
      <li>Halving arithmetic precision (FP32 → INT8) does not reduce DRAM traffic for memory-bound sparse workloads.</li>
      <li>As NPU compilers mature toward explicit mixed-precision support, operator coverage should expand; the memory-bandwidth bottleneck will persist for any streaming-dataflow architecture without sparse acceleration.</li>
    </ul>
  </div>
</div>

<Glossary :terms="['npu', 'cpu-fallback', 'int8', 'scatter-gather']" />

---
layout: default
---

## Scientific Limitations & Threats to Validity
### Experimental Constraints

<div class="grid-cols-2 mt-2">
  <div>
    <ul>
      <li><strong>Single platform:</strong> Results restricted to Core Ultra 5 125H. Memory SKU variations affect bandwidth.</li>
      <li><strong>Power telemetry:</strong> SoCWatch cannot isolate NPU rail — Meteor Lake PMT counters lack an NPU power/energy rail definition. Package-level CPU/GPU only.</li>
      <li><strong>Future work:</strong> Shunt resistors &amp; oscilloscope for isolated NPU power measurement.</li>
    </ul>
  </div>

  <div>
    <ul>
      <li><strong>Energy model:</strong> <KaTeX math="E = P \times t" /> assumes stationary power; background OS activity not isolated.</li>
      <li><strong>Statistics:</strong> Bootstrap 95% CIs per config; paired hypothesis tests across devices pending.</li>
    </ul>
  </div>
</div>

<Glossary :terms="['socwatch', 'warm-up', 'onnx']" />

---
layout: default
---

## Practical Guidance for Edge AI Deployments
### System Guidelines and Future Roadmap

<div class="mt-2 flex flex-col gap-2">
  <div class="highlight-box highlight-box-success">
    <div class="highlight-box-title">✔ Recommended: Dense Vision on NPU (FP32)</div>
    <div class="text-xs">
      Deploy vision networks (ResNet50, MobileNetV2) and regular transformers to the <strong>NPU</strong> at <strong>FP32</strong> (4.5–11.4× vs CPU). Avoid INT8 for vision on NPU — toolchain silently falls back to CPU.
    </div>
  </div>

  <div class="highlight-box highlight-box-warning">
    <div class="highlight-box-title">⚠ Caution: GNN Workloads on GPU (FP32)</div>
    <div class="text-xs">
      Deploy GNNs (GCN, GraphSAGE, GraphTransformer) to the <strong>iGPU</strong> at <strong>FP32</strong>. Avoid NPU due to compilation failures, memory bottlenecks, and silent fallbacks (OpenVINO 2024.1).
    </div>
  </div>
</div>

---
layout: default
---

## Key Findings Summary
### What This Study Reveals

<div class="grid-cols-2 mt-2">
  <div class="glass-panel" style="border-left:3px solid var(--color-emerald)">
    <h3 class="text-emerald font-bold" style="font-size:0.8rem">✅ NPU Excels For</h3>
    <ul class="text-xs" style="margin:0">
      <li>Dense vision at <strong>FP32</strong> (4.5–11.4× vs CPU)</li>
      <li>Structured attention (ViT: 11.4×)</li>
    </ul>
  </div>
  <div class="glass-panel" style="border-left:3px solid var(--color-rose)">
    <h3 class="text-rose font-bold" style="font-size:0.8rem">❌ NPU Struggles With</h3>
    <ul class="text-xs" style="margin:0">
      <li>GNNs: CPU parity (±6%), no acceleration</li>
      <li>INT8: 0.45–1.21× (SGC: 2.2× regression)</li>
    </ul>
  </div>
  <div class="glass-panel" style="border-left:3px solid var(--color-blue)">
    <h3 class="text-blue font-bold" style="font-size:0.8rem">📊 Best GNN Backend</h3>
    <ul class="text-xs" style="margin:0">
      <li><strong>iGPU</strong> — lowest GNN latency</li>
      <li>Power: 9.1–12.5 W; INT8 energy model-dependent</li>
    </ul>
  </div>
  <div class="glass-panel" style="border-left:3px solid var(--color-amber)">
    <h3 class="text-amber font-bold" style="font-size:0.8rem">🔧 Toolchain Issues</h3>
    <ul class="text-xs" style="margin:0">
      <li>Silent CPU fallback for quantized models</li>
      <li>Static-shape decouples latency from sparsity</li>
    </ul>
  </div>
</div>

<Glossary :terms="['npu', 'igpu', 'int8', 'cpu-fallback', 'static-shape']" />

---
layout: default
---

## Key References
### Selected Bibliography

<div class="grid-cols-2 mt-2">
  <div>
    <ul style="font-size:0.7rem">
      <li><strong>Meteor Lake</strong> — Intel Hot Chips 2023, Foveros 3D packaging</li>
      <li><strong>GNN Accelerators</strong> — HyGCN (ISPASS 2020), EnGN (ISCA 2020), GRIP (HPCA 2021)</li>
      <li><strong>OpenVINO NPU</strong> — Intel NPU plugin, operator coverage, IR pipeline</li>
      <li><strong>MLPerf Inference</strong> — Standardized benchmark suite</li>
    </ul>
  </div>
  <div>
    <ul style="font-size:0.7rem">
      <li><strong>OGB Datasets</strong> — Hu et al., NeurIPS 2020</li>
      <li><strong>Roofline on Edge</strong> — Bi et al., arXiv 2026</li>
      <li><strong>GNN Architectures</strong> — GCN (Kipf 2017), GAT (2018), GraphSAGE (2017), MPNN (2017)</li>
    </ul>
  </div>
</div>

<Glossary :terms="['openvino', 'gnn', 'npu']" />

---
layout: default
---

<div class="flex flex-col items-center justify-center h-full gap-5">
  <h1 class="text-3xl font-extrabold text-blue-800 text-center">Thank You!</h1>
  <h3 class="text-xl text-slate-600 font-semibold">Questions &amp; Discussion</h3>
  
  <div class="flex items-start gap-5 p-5 bg-slate-50 border border-slate-200 rounded-lg shadow-sm max-w-xl w-full mt-1">
    <div class="flex-1">
      <div class="font-bold text-base text-slate-900 text-center">
        Yusuf Talha ARABACI, Emrullah DEMİRAL, Ömer Faruk ACAR
      </div>
      <div class="text-slate-700 text-xs mt-1 text-center font-medium">
        Department of Software Engineering, Karabük University
      </div>
      <div class="mt-3 text-xs text-slate-500 text-center leading-relaxed">
        <strong>Paper &amp; Data:</strong><br />
        <a href="https://github.com/yusufarbc/intel-npu-gnn-benchmarking" target="_blank" class="text-blue-600 underline font-medium">
          github.com/yusufarbc/intel-npu-gnn-benchmarking
        </a>
      </div>
      <div class="mt-1 text-xs text-slate-500 text-center">
        <strong>Contact:</strong> yusuftalhaarabaci@hotmail.com
      </div>
    </div>
    <div class="flex-shrink-0 text-center pt-1">
      <img src="./public/qrcode.png" class="w-28 h-28 object-contain rounded border border-slate-200 shadow-sm" alt="QR Code - GitHub Repository" />
      <div class="text-xs text-slate-400 mt-1">Scan me</div>
    </div>
  </div>
</div>

<Glossary :terms="['npu', 'igpu', 'int8', 'fp32']" />
