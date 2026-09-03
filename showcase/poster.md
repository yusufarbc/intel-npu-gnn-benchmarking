---
theme: default
layout: poster
terms: [npu, gnn, igpu, fp32, meteor-lake]
info: |
  ## Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis
  IEEE HPEC 2026 interactive poster.
class: text-left
highlighter: shiki
drawings:
  persist: false
transition: fade
routerMode: hash
title: "Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis"
---

<div class="flex flex-col h-full justify-between">
  <div>
    <div class="text-sm font-semibold text-blue-700 tracking-wide">IEEE HPEC 2026 · GRAPH ANALYTICS AND APPLICATION BENCHMARKING</div>
    <h1 class="text-3xl font-extrabold text-slate-900 leading-tight mt-3">
      Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis
    </h1>
    <div class="text-xl text-slate-500 mt-2">Measurements on a power-constrained Meteor Lake client platform</div>
  </div>

  <div class="grid grid-cols-12 gap-8 items-center">
    <div class="col-span-9">
      <div class="text-2xl leading-snug font-semibold text-slate-800 border-l-8 border-blue-700 pl-5">
        The NPU accelerates regular FP32 vision models, but the iGPU is generally the better accelerator for the sparse GNN workloads evaluated here.
      </div>
      <div class="text-base text-slate-600 mt-6">
        Yusuf Talha Arabacı · Emrullah Demiral · Ömer Faruk Acar<br/>
        Department of Software Engineering, Karabük University
      </div>
    </div>
    <div class="col-span-3 text-center mb-7">
      <a href="https://github.com/yusufarbc/intel-npu-gnn-benchmarking" target="_blank" rel="noopener noreferrer">
        <img src="/qrcode.png" class="w-36 h-36 mx-auto border border-slate-200 rounded-lg hover:shadow-md transition-shadow" alt="QR code linking to the GitHub repository" />
      </a>
      <div class="text-xs text-slate-500 mt-2">Paper · code · data · figures</div>
    </div>
  </div>
</div>

<!--
[Sources]
- paper/paper.tex, abstract and Sections I, IV, and VI.
- results/figures/comparison_table.csv.
-->

---
layout: poster
terms: [npu, gnn, spmm, scatter-gather, memory-bound]
---

## Why sparse GNNs challenge dense accelerators

<div class="grid grid-cols-2 gap-10 mt-6 items-start">
  <div>
    <h3 class="text-xl font-bold text-emerald-700">What the NPU favors</h3>
    <ul class="text-lg leading-relaxed mt-3">
      <li>Regular tensor shapes</li>
      <li>Predictable data movement</li>
      <li>Dense Conv and MatMul kernels</li>
      <li>Long fusion-friendly operator chains</li>
    </ul>
  </div>
  <div>
    <h3 class="text-xl font-bold text-rose-700">What GNN inference introduces</h3>
    <ul class="text-lg leading-relaxed mt-3">
      <li>Sparse-times-dense multiplication (SpDMM)</li>
      <li>Gather, Scatter, and indirect indexing</li>
      <li>Irregular operator structure and data movement</li>
      <li>Operator and device-assignment uncertainty</li>
    </ul>
  </div>
</div>

<div class="mt-10 text-center text-xl font-semibold text-slate-800">
  Research question: when does a client NPU provide real end-to-end benefit rather than nominal device availability?
</div>

<!--
[Sources]
- paper/paper.tex, Sections I and II.
- Abadal et al., Computing Graph Neural Networks: A Survey from Algorithms to Accelerators, DOI 10.1145/3477141.
-->

---
layout: poster
terms: [igpu, ogb, openvino, warm-up, socwatch]
---

## Measurement setup and protocol

<div class="grid grid-cols-12 gap-8 mt-5">
  <div class="col-span-5">
    <img src="/meteor-lake-architecture.jpg" class="w-full max-h-80 object-contain rounded-lg border border-slate-200" alt="Intel Meteor Lake platform architecture" />
  </div>
  <div class="col-span-7">
    <table class="w-full text-base">
      <tbody>
        <tr><th class="text-left">Platform</th><td>Core Ultra 5 125H, 16 GB LPDDR5x, Windows 11</td></tr>
        <tr><th class="text-left">Backends</th><td>14-core CPU · Arc iGPU · AI Boost NPU 3720</td></tr>
        <tr><th class="text-left">Workloads</th><td>9 GNNs + 5 dense CNN/transformer baselines</td></tr>
        <tr><th class="text-left">Graph inputs</th><td>Fixed 2,708-node, 10,000-edge tensors derived from three OGB source graphs</td></tr>
        <tr><th class="text-left">Protocol</th><td>Batch 1 · 5 warm-ups · 100 iterations · 3 runs</td></tr>
        <tr><th class="text-left">Paper stack</th><td>OpenVINO 2024.1 · ONNX Runtime 1.18</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="mt-6 text-base text-slate-600">
  Timed latency includes provider dispatch and required host-device transfers after warm-up. Loading, preprocessing, conversion, and one-time compilation are excluded.
</div>

<!--
[Sources]
- paper/paper.tex, Section III and Tables I-II.
- Gomes et al., Meteor Lake and Arrow Lake, IEEE Hot Chips 34, 2022, DOI 10.1109/HCS55958.2022.9895532.
-->

---
layout: poster
terms: [npu, igpu, fp32, mobilenet, vit]
---

## Selected dense models benefit; the iGPU leads most evaluated GNNs

<div class="grid grid-cols-12 gap-6 mt-3 items-center">
  <div class="col-span-8">
    <img src="/figures/fig1_latency_comparison.svg" class="w-full max-h-96 object-contain" alt="FP32 latency comparison across CPU, iGPU, and NPU" />
  </div>
  <div class="col-span-4 text-lg leading-relaxed">
    <div><strong class="text-emerald-700">MobileNetV2:</strong><br/>1.90 ms on NPU<br/><strong>4.5x vs. CPU</strong></div>
    <div class="mt-5"><strong class="text-emerald-700">ViT-Tiny:</strong><br/>9.10 ms on NPU<br/><strong>11.4x vs. CPU</strong></div>
    <div class="mt-5"><strong class="text-blue-700">GraphTransformer:</strong><br/>6.03 ms iGPU<br/>10.72 ms NPU</div>
    <div class="mt-5 text-sm text-slate-600">Backend choice remains model-dependent; iGPU is faster for ViT-Tiny and EfficientNet-B0.</div>
  </div>
</div>

<div class="text-sm text-slate-500 mt-2 text-center">Mean FP32 latency; GNN inputs are OGB-derived fixed tensors, while dense baselines repeat the same synthetic input. Lower is better.</div>

<!--
[Sources]
- paper/paper.tex, Section IV-A and Fig. 1.
- results/figures/comparison_table.csv.
-->

---
layout: poster
terms: [int8, cpu-fallback, dynamic-quantization, sgc, mpnn]
---

## INT8 is a compatibility decision, not a guaranteed speedup

<div class="grid grid-cols-12 gap-7 mt-3 items-center">
  <div class="col-span-7">
    <img src="/figures/fig2_int8_speedup_heatmap.svg" class="w-full max-h-80 object-contain" alt="INT8 speedup heatmap" />
    <div class="text-sm text-slate-500 text-center">Speedup below 1.0 means INT8 is slower than FP32.</div>
  </div>
  <div class="col-span-5">
    <table class="w-full text-base">
      <thead><tr><th>Model</th><th>Requested</th><th>Outcome</th></tr></thead>
      <tbody>
        <tr><td>SGC</td><td>NPU INT8</td><td><strong>2.2x slower</strong></td></tr>
        <tr><td>MPNN</td><td>NPU FP32</td><td>No retained native-NPU result</td></tr>
        <tr><td>MobileNetV2</td><td>NPU INT8</td><td>CPU-like timing; placement unverified</td></tr>
        <tr><td>GAT / GATv2</td><td>NPU INT8</td><td>No executable configuration</td></tr>
        <tr><td>EfficientNet-B0</td><td>NPU INT8</td><td>No executable configuration</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="mt-2 text-lg font-semibold text-slate-800 text-center">
  Retain and inspect per-operator assignment evidence: a requested NPU configuration does not prove native NPU execution.
</div>

<!--
[Sources]
- paper/paper.tex, Sections IV-B and IV-D, Table III, and Table IV.
- results/figures/summary_stats.csv.
-->

---
layout: poster
terms: [npu, igpu, fp32, int8]
---

## Deployment guidance—and the boundary of the evidence

<div class="grid grid-cols-2 gap-10 mt-5">
  <div>
    <h3 class="text-xl font-bold text-blue-700">Use the measurements this way</h3>
    <ul class="text-lg leading-relaxed mt-3">
      <li><strong>Dense FP32:</strong> benchmark both NPU and iGPU; the winner is model-dependent</li>
      <li><strong>Evaluated GNNs:</strong> start with the iGPU</li>
      <li><strong>INT8:</strong> benchmark per model and verify assignment</li>
      <li><strong>Energy:</strong> do not infer isolated NPU power from package telemetry</li>
    </ul>
  </div>
  <div>
    <h3 class="text-xl font-bold text-slate-700">Do not overgeneralize</h3>
    <ul class="text-lg leading-relaxed mt-3">
      <li>One Core Ultra 5 125H system</li>
      <li>One frozen paper software stack</li>
      <li>Established GNN reference workloads</li>
      <li>Fixed-shape subgraphs, not full-graph OGB inference</li>
      <li>No post-quantization accuracy study</li>
      <li>Latency-derived energy is a heuristic estimate</li>
    </ul>
  </div>
</div>

<div class="grid grid-cols-12 gap-6 items-center mt-8 border-t border-slate-200 pt-5">
  <div class="col-span-9 text-xl font-semibold text-slate-800">
    Explore the paper, code, CSV results, and reproduction workflow in the repository.<br/>
    <a href="https://github.com/yusufarbc/intel-npu-gnn-benchmarking" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 hover:underline font-normal text-base inline-block mt-2">
      https://github.com/yusufarbc/intel-npu-gnn-benchmarking ↗
    </a>
  </div>
  <div class="col-span-3 text-center">
    <a href="https://github.com/yusufarbc/intel-npu-gnn-benchmarking" target="_blank" rel="noopener noreferrer">
      <img src="/qrcode.png" class="w-28 h-28 mx-auto hover:opacity-90 transition-opacity" alt="QR code linking to the GitHub repository" />
    </a>
  </div>
</div>

<!--
[Sources]
- paper/paper.tex, Sections V-VI.
- README.md, interpretation boundaries and environment notes.
-->
