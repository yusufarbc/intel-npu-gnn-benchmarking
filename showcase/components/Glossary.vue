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
    desc: 'An integrated circuit design that combines all core computing blocks (CPU, GPU, NPU, memory controller, media engine) onto a single silicon chip.'
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
    desc: 'A compiler mechanism where unsupported operators in a model compiled for NPU or GPU are dynamically and silently routed to the CPU, severely degrading latency.'
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
    desc: 'Intel\'s low-level system profiling tool that reads hardware telemetry (via PMT) to monitor package, CPU, iGPU, and NPU power usage (mW/mJ).'
  },
  'throughput-watt': {
    title: 'Throughput per Watt',
    desc: 'An energy efficiency metric expressing the number of inferences, nodes, or edges processed per Watt of power consumed.'
  },
  'warm-up': {
    title: 'Warm-up Iterations',
    desc: 'Non-timed execution runs executed at startup to load libraries and populate caches, preventing cold-start latency from skewing benchmark statistics.'
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
  position: absolute;
  bottom: 0.65rem;
  left: 20rem; /* Positions next to the footer-text (which is at left: 2rem) */
  z-index: 1000;
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
  left: 0;
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
