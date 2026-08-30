<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  terms: {
    type: Array,
    required: true
  }
})

const isOpen = ref(false)

const glossary = {
  'npu': {
    title: 'NPU (Neural Processing Unit)',
    desc: 'A specialized hardware co-processor/ASIC designed to accelerate tensor and matrix multiplication (MAC) operations with high energy efficiency. E.g., Intel AI Boost NPU.'
  },
  'igpu': {
    title: 'iGPU (Integrated GPU)',
    desc: 'A graphics processor integrated onto the same silicon die as the CPU. The Intel Arc iGPU (7 Xe-cores) shares the LPDDR5x fabric and often outperforms the NPU on GNNs due to higher memory bandwidth.'
  },
  'meteor-lake': {
    title: 'Meteor Lake / Core Ultra',
    desc: 'Intel\'s disaggregated tile-based CPU architecture utilizing Foveros 3D packaging to combine CPU, Graphics, SoC, and NPU tiles into a single processor package.'
  },
  'soc': {
    title: 'SoC (System on Chip)',
    desc: 'An integrated circuit design that combines all core computing blocks (CPU, iGPU, NPU, memory controller, media engine) into one package or tightly integrated platform.'
  },
  'compute-bound': {
    title: 'Compute Bound',
    desc: 'A performance bottleneck where execution speed is limited by the processor\'s arithmetic throughput (FLOPS/MACs), typical for dense models like CNNs (ResNet50).'
  },
  'memory-bound': {
    title: 'Memory Bound',
    desc: 'A performance bottleneck where execution speed is limited by DRAM bandwidth/latency (moving data between memory and registers) rather than arithmetic power. GNNs are highly memory-bound.'
  },
  'stalls': {
    title: 'Stall Cycles',
    desc: 'CPU or NPU clock cycles wasted while execution units sit idle waiting for data to arrive from main memory (DRAM) due to cache misses.'
  },
  'locality': {
    title: 'Data/Cache Locality',
    desc: 'The degree of data proximity in space (spatial) or reuse in time (temporal). GNNs exhibit poor cache locality due to irregular graph structures.'
  },
  'gnn': {
    title: 'GNN (Graph Neural Network)',
    desc: 'A class of deep learning architectures designed to process non-Euclidean, unstructured graph data by operating directly on node features and their connections.'
  },
  'sparse-graph': {
    title: 'Sparse Graph',
    desc: 'A graph where the number of actual edges is extremely low compared to the theoretical maximum (e.g., ogbn-arxiv, with an average degree of 6.9).'
  },
  'dense-graph': {
    title: 'Dense Graph',
    desc: 'A graph with high connectivity where node connections approach the theoretical maximum (e.g., ogbn-proteins, average degree of 451.7).'
  },
  'adjacency-matrix': {
    title: 'Adjacency Matrix',
    desc: 'An N x N matrix representing node connectivity in a graph. In sparse GNN workloads, this matrix contains mostly zero elements.'
  },
  'spmm': {
    title: 'SpMM (Sparse-Dense Matrix Multiplication)',
    desc: 'A core GNN operation multiplying a sparse adjacency matrix by a dense vertex feature matrix, requiring irregular, indirect memory lookups.'
  },
  'message-passing': {
    title: 'Message Passing / Neighborhood Aggregation',
    desc: 'The fundamental GNN computation step where each node updates its representation by aggregating (summing, averaging) feature vectors from its neighbors.'
  },
  'irregular-access': {
    title: 'Irregular Memory Access',
    desc: 'Non-sequential, data-dependent memory read/write strides determined by graph connectivity, which cause cache thrashing and invalidate standard prefetchers.'
  },
  'power-law': {
    title: 'Power-Law Distribution',
    desc: 'A topological property of real-world graphs where a tiny fraction of nodes (hubs) have extremely high degrees, while the vast majority have very few connections.'
  },
  'openvino': {
    title: 'OpenVINO',
    desc: 'Intel\'s open-source inference engine and compiler toolchain that optimizes neural networks (ONNX, PyTorch) for Intel CPUs, iGPUs, and NPUs.'
  },
  'operator-fusion': {
    title: 'Operator Fusion',
    desc: 'A compiler optimization combining consecutive graph operations (e.g., MatMul + Bias + ReLU) into a single execution kernel to reduce intermediate DRAM traffic.'
  },
  'fgr': {
    title: 'FGR (Fusion Gain Ratio)',
    desc: 'A metric measuring the latency improvement achieved by applying compiler-level operator fusion passes.'
  },
  'cei': {
    title: 'CEI (Compilation Efficiency Index)',
    desc: 'A benchmark index correlating a model\'s theoretical operations (FLOPs) with its physical execution latency on the hardware.'
  },
  'cpu-fallback': {
    title: 'CPU Fallback',
    desc: 'A compiler mechanism where unsupported operators requested for an accelerator such as the NPU or iGPU execute on the CPU; this can change the meaning and latency of a device comparison.'
  },
  'ir-conversion': {
    title: 'IR Conversion',
    desc: 'The process of compiling an ONNX/PyTorch model into OpenVINO\'s Intermediate Representation format (.xml for topology and .bin for weight parameters).'
  },
  'fp32': {
    title: 'FP32 (32-bit Floating Point)',
    desc: 'Single-precision floating-point format used for training and high-accuracy inference. It provides high precision but has a larger memory footprint.'
  },
  'int8': {
    title: 'INT8 Quantization',
    desc: 'A post-training optimization mapping 32-bit floating-point weights and activations to 8-bit integers, reducing memory footprint by 4x and accelerating MAC operations.'
  },
  'regression': {
    title: 'Performance Regression',
    desc: 'An optimization failure where applying a technique (like INT8 quantization) degrades execution latency, typically due to quantization/dequantization overhead or memory bound limitations (e.g., SGC).'
  },
  'intensity': {
    title: 'Arithmetic Intensity',
    desc: 'The ratio of computational operations (FLOPs) performed per byte of memory transferred. GNNs have very low intensity, while CNNs have high intensity.'
  },
  'structured-attention': {
    title: 'Structured Attention',
    desc: 'Self-attention calculated over fixed, contiguous grid structures (e.g., Vision Transformers), matching the spatial streaming optimizations of the NPU perfectly.'
  },
  'irregular-attention': {
    title: 'Irregular Attention',
    desc: 'Self-attention calculated over dynamic, non-contiguous graph neighborhoods (e.g., GraphTransformer), causing cache misses and compiler fusion limits.'
  },
  'spectral-spatial': {
    title: 'Spectral vs. Spatial GNN',
    desc: 'Spectral GNNs (GCN) approximate filters via Graph Fourier transforms, while Spatial GNNs (GraphSAGE) aggregate direct neighborhood connections.'
  },
  'socwatch': {
    title: 'SoCWatch',
    desc: 'Intel\'s low-level system profiling tool that reads hardware telemetry (via PMT) to monitor package, CPU, and iGPU power usage (mW/mJ). The evaluated Meteor Lake PMT interface did not expose an isolated NPU power rail.'
  },
  'throughput-watt': {
    title: 'Throughput per Watt',
    desc: 'An energy efficiency metric expressing the number of inferences, nodes, or edges processed per Watt of power consumed.'
  },
  'warm-up': {
    title: 'Warm-up Iterations',
    desc: 'Non-timed execution runs executed at startup to load libraries and populate caches, preventing cold-start latency from skewing benchmark statistics.'
  },
  'onnx': {
    title: 'ONNX (Open Neural Network Exchange)',
    desc: 'An open standard format for representing ML models. All 14 models were exported to ONNX at FP32; INT8 variants used ONNX Runtime\'s dynamic quantization API.'
  },
  'lpddr5x': {
    title: 'LPDDR5x',
    desc: 'Low-power double data rate 5X DRAM used by Meteor Lake (~120 GB/s peak bandwidth). GNNs are memory-bound by this bandwidth rather than compute capacity.'
  },
  'ogb': {
    title: 'OGB (Open Graph Benchmark)',
    desc: 'A standardized suite of large-scale graph datasets. The paper uses ogbn-arxiv (citation), ogbn-products (co-purchase), and ogbn-proteins (protein interactions).'
  },
  'roofline': {
    title: 'Roofline Model',
    desc: 'An analytical model plotting achieved throughput against arithmetic intensity to identify whether a workload is memory-bound or compute-bound on a given architecture.'
  },
  'static-shape': {
    title: 'Static-Shape Compilation',
    desc: 'An ONNX/OpenVINO mode where input tensor dimensions are fixed at compile time. Causes NPU latency to be constant regardless of input graph sparsity (r ≈ -0.00 correlation with edges/node).'
  },
  'scatter-gather': {
    title: 'Scatter / Gather',
    desc: 'Indexed memory operations reading from or writing to non-contiguous locations. Heavily used by GNN attention/aggregation; unsupported in NPU\'s quantized operator set, causing INT8 failures.'
  },
  'vit': {
    title: 'ViT (Vision Transformer)',
    desc: 'A transformer applied to image patches on a regular 2D grid. ViT-Tiny achieves 11.4× NPU speedup over CPU because its attention patterns are static and predictable.'
  },
  'xe-lpg': {
    title: 'Xe-LPG',
    desc: 'Intel\'s integrated GPU microarchitecture in the evaluated Meteor Lake system (7 Xe-cores). Its memory system and measured optimization behavior are plausible contributors to its lower latency on the tested GNNs, but the experiment does not isolate one cause.'
  },
  'tdp': {
    title: 'TDP (Thermal Design Power)',
    desc: 'A thermal-design target for a processor platform; it is not the same as directly measured CPU, iGPU, or NPU power during a benchmark.'
  },
  'ort': {
    title: 'ONNX Runtime (ORT)',
    desc: 'An open-source cross-platform engine designed to run machine learning models efficiently. ONNX Runtime traces and quantization APIs were used for model execution.'
  },
  'bootstrap': {
    title: 'Bootstrap Resampling',
    desc: 'A non-parametric statistical method estimating the distribution of sample statistics (like mean latency or 95% confidence intervals) by repeatedly resampling the observed metrics with replacement.'
  },
  'foveros': {
    title: 'Foveros 3D Packaging',
    desc: 'Intel\'s 3D packaging technology, used in Meteor Lake to integrate compute, graphics, SoC, and I/O tiles.'
  },
  'pmt': {
    title: 'PMT (Platform Monitoring Technology)',
    desc: 'Intel\'s hardware telemetry technology providing low-level access to system energy, temperature, and performance counters (used by SoCWatch).'
  },
  'dvfs': {
    title: 'DVFS (Dynamic Voltage and Frequency Scaling)',
    desc: 'A power management technique in modern processors that adjusts frequency and voltage dynamically based on workload and temperature, affecting run-to-run latency consistency.'
  },
  'ptq': {
    title: 'PTQ (Post-Training Quantization)',
    desc: 'A static quantization method converting weights and activations to INT8 using a fixed calibration dataset before deployment.'
  },
  'dynamic-quantization': {
    title: 'Dynamic Quantization',
    desc: 'An optimization method where activation scale factors are computed dynamically at runtime, removing calibration dataset requirements but introducing runtime overhead.'
  },
  'lpe-core': {
    title: 'LPE-core (Low Power Efficient Core)',
    desc: 'Low Power Efficient CPU cores located on the Meteor Lake SoC tile designed to handle background tasks with minimal energy consumption.'
  },
  'thermal-throttling': {
    title: 'Thermal Throttling',
    desc: 'A hardware protection mechanism that reduces clock speed when temperature thresholds are exceeded, potentially introducing latency variances during long benchmarking runs.'
  },
  'sgc': {
    title: 'SGC (Simple Graph Convolution)',
    desc: 'A simplified GNN architecture that removes non-linearities and collapses consecutive weight matrices, highlighting the runtime dispatch overhead when quantized.'
  },
  'gcn': {
    title: 'GCN (Graph Convolutional Network)',
    desc: 'A semi-supervised spectral GNN architecture approximating first-order graph spectral convolutions via localized neighborhood averaging.'
  },
  'gat': {
    title: 'GAT (Graph Attention Network)',
    desc: 'An attentional GNN that computes dynamic edge weights via self-attention over node neighborhoods, which often fails INT8 NPU compilation due to irregular attention graphs.'
  },
  'flops': {
    title: 'FLOPs / FLOPS',
    desc: 'Floating Point Operations (or per Second), measuring the theoretical computational complexity of a model or the processing throughput of a backend.'
  },
  'dram': {
    title: 'DRAM (Dynamic RAM)',
    desc: 'Main system memory. High latency and memory bandwidth limitations of DRAM form the primary bottleneck for sparse, irregular GNN workloads.'
  },
  'appnp': {
    title: 'APPNP (Approximate Personalized Propagation of Neural Predictions)',
    desc: 'A GNN propagation model based on personalized PageRank, decoupling feature transformation from propagation to achieve scalability.'
  },
  'gin': {
    title: 'GIN (Graph Isomorphic Network)',
    desc: 'An expressive spatial GNN designed to model graph isomorphism and achieve the same discriminative power as the Weisfeiler-Lehman (1-WL) graph test.'
  },
  'graphsage': {
    title: 'GraphSAGE (Sample and Aggregate)',
    desc: 'A spatial GNN framework that aggregates node features from a fixed-size sampled local neighborhood rather than the full neighborhood.'
  },
  'mpnn': {
    title: 'MPNN (Message Passing Neural Network)',
    desc: 'A general GNN framework that generalizes various models using message creation, aggregation, and node update phases.'
  },
  'bert': {
    title: 'BERT (Bidirectional Encoder Representations from Transformers)',
    desc: 'A transformer-based NLP model. The paper evaluates BERT-Tiny to benchmark attention patterns on the NPU and CPU.'
  },
  'resnet': {
    title: 'ResNet50',
    desc: 'A deep 50-layer convolutional neural network leveraging residual shortcuts, serving as a dense baseline for spatial streaming comparisons.'
  },
  'mobilenet': {
    title: 'MobileNetV2',
    desc: 'A lightweight CNN architecture optimized for edge devices utilizing inverted residual blocks and linear bottlenecks.'
  },
  'etw': {
    title: 'ETW (Event Tracing for Windows)',
    desc: 'A kernel-level tracing facility provided by Windows, used by Intel SoCWatch to capture low-level platform events.'
  }
}

const activeTerms = computed(() => {
  return props.terms.map(key => glossary[key]).filter(Boolean)
})
</script>

<template>
  <div class="glossary-wrapper">
    <!-- Trigger Button -->
    <button 
      @click="isOpen = !isOpen"
      class="glossary-btn"
      title="Click to view technical term explanations"
    >
      <span class="icon">💡</span>
      <span class="text">Term Notes</span>
    </button>

    <!-- Dropup Drawer Modal -->
    <div v-if="isOpen" class="glossary-modal">
      <div class="glossary-header">
        <h4 class="title">📖 Technical Explanations</h4>
        <button @click="isOpen = false" class="close-btn">&times;</button>
      </div>
      <div class="glossary-content">
        <div v-for="item in activeTerms" :key="item.title" class="glossary-item">
          <h5 class="item-title">{{ item.title }}</h5>
          <p class="item-desc">{{ item.desc }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.glossary-wrapper {
  /* Keep the interactive control above full-slide figures/tables and make
     Slidev's browser and PDF renderers place it consistently on every page. */
  position: fixed;
  bottom: 0.65rem;
  right: 5.5rem; /* Position next to the page number (which is at right: 2rem) */
  z-index: 2147483647;
  isolation: isolate;
}

.glossary-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  background-color: var(--color-slate-100);
  border: 1px solid var(--color-slate-300);
  color: var(--color-blue);
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.65rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
}

.glossary-btn:hover {
  background-color: var(--color-slate-200);
  border-color: var(--color-blue);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.glossary-modal {
  position: absolute;
  bottom: 1.8rem;
  right: 0; /* Align right edge of the modal with the button */
  width: 320px;
  max-height: 280px;
  background: #ffffff;
  border: 1px solid var(--color-slate-300);
  border-radius: 6px;
  box-shadow: 0 -4px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  animation: slideUp 0.18s ease-out;
}

.glossary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: var(--color-slate-50);
  border-bottom: 1px solid var(--color-slate-200);
  padding: 0.4rem 0.6rem;
}

.glossary-header .title {
  margin: 0;
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--color-slate-900);
  font-family: 'Outfit', sans-serif;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1rem;
  color: var(--color-slate-500);
  cursor: pointer;
  padding: 0;
  line-height: 1;
}

.close-btn:hover {
  color: var(--color-rose);
}

.glossary-content {
  overflow-y: auto;
  padding: 0.5rem;
  text-align: left;
}

.glossary-item {
  margin-bottom: 0.5rem;
  border-bottom: 1px solid var(--color-slate-100);
  padding-bottom: 0.4rem;
}

.glossary-item:last-child {
  margin-bottom: 0;
  border-bottom: none;
  padding-bottom: 0;
}

.item-title {
  margin: 0 0 0.15rem 0;
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--color-blue);
  font-family: 'Outfit', sans-serif;
}

.item-desc {
  margin: 0;
  font-size: 0.65rem;
  color: var(--color-slate-700);
  line-height: 1.3;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
