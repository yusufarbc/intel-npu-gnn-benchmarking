"""
Organizes paper/references/ directory:
  1. Deletes scratch / personal files
  2. Renames .md files to "NN - AuthorYEAR - Title.md"
  3. Rewrites paper/references/index.md
  4. Generates docs/references.md

Run: python analysis/organize_references.py [--dry-run]
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
REFS_DIR  = REPO_ROOT / "paper" / "references"
DOCS_DIR  = REPO_ROOT / "docs"

# Non-academic files to remove
JUNK_FILES = {
    "topic.txt",
    "gnn.txt",
    "list.txt",
    "regular-list.txt",
    "references",
    "pdf_processor.py",
    "list_pdf_hashes.py",
    "GNN_NPU_Literatur_Tarama_Rehberi.docx",
    "GNN_NPU_Literatur_Tarama_Rehberi.txt",
    "Intel NPU'da GNN Performans Analizi.docx",
    "Intel NPU'da GNN Performans Analizi.txt",
    "academic-sources-summary.md",
    "23 - Corporate PowerPoint Template Use IntelOne Fonts For All Text  (General Employee Usage).md",
    "23 - Corporate PowerPoint Template Use IntelOne Fonts For All Text  (General Employee Usage).pdf",
}

# ---------------------------------------------------------------------------
CANONICAL = [
    (1,  "Kipf2017",       "Semi-Supervised Classification with Graph Convolutional Networks"),
    (2,  "Velickovic2018", "Graph Attention Networks"),
    (3,  "Brody2022",      "How Attentive are Graph Attention Networks (GATv2)"),
    (4,  "Hamilton2017",   "Inductive Representation Learning on Large Graphs (GraphSAGE)"),
    (5,  "Xu2019",         "How Powerful are Graph Neural Networks (GIN)"),
    (6,  "Wu2019",         "Simplifying Graph Convolutional Networks (SGC)"),
    (7,  "Gasteiger2019",  "Predict then Propagate - Graph Neural Networks meet Personalized PageRank (APPNP)"),
    (8,  "Gilmer2017",     "Neural Message Passing for Quantum Chemistry (MPNN)"),
    (9,  "Dwivedi2021",    "A Generalization of Transformer Networks to Graphs (GraphTransformer)"),
    (10, "He2016",         "Deep Residual Learning for Image Recognition (ResNet)"),
    (11, "Sandler2018",    "MobileNetV2 Inverted Residuals and Linear Bottlenecks"),
    (12, "Tan2019",        "EfficientNet Rethinking Model Scaling for Convolutional Neural Networks"),
    (13, "Dosovitskiy2021","An Image is Worth 16x16 Words - ViT"),
    (14, "Turc2019",       "Well-Read Students Learn Better On the Importance of Pre-training Compact Models (BERT-tiny)"),
    # ── Datasets ─────────────────────────────────────────────────────────
    (15, "Hu2020",         "Open Graph Benchmark - Datasets for Machine Learning on Graphs"),
    # ── GNN Hardware Accelerators ─────────────────────────────────────────
    (16, "Auten2020",      "Hardware Acceleration of Graph Neural Networks"),
    (17, "Liang2021",      "EnGN A High-Throughput and Energy-Efficient Accelerator for Large Graph Neural Networks"),
    (18, "Yan2020",        "HyGCN A GCN Accelerator with Hybrid Architecture"),
    (19, "Kiningham2023",  "GRIP A Graph Neural Network Accelerator Architecture"),
    (20, "Abadal2021",     "Computing Graph Neural Networks A Survey from Algorithms to Accelerators"),
    (21, "Zhang2026",      "A Survey on Graph Neural Network Acceleration Algorithms Systems and Customized Hardware"),
    (22, "GNNMark2021",    "GNNMark A Benchmark Suite to Characterize Graph Neural Network Training on GPUs"),
    # ── Intel NPU and Hardware Architecture ──────────────────────────────
    (23, "Jouppi2017",     "In-Datacenter Performance Analysis of a Tensor Processing Unit"),
    (24, "Intel2024",      "Heterogeneous AI Powerhouse - Intel Core Ultra NPU Architecture"),
    (25, "Lam2024",        "Intel Meteor Lake NPU - Architecture Analysis"),
    # ── Edge AI and NPU Inference ────────────────────────────────────────
    (26, "Xu2025",         "Fast On-device LLM Inference with NPUs"),
    (27, "LLM_NPU2025",    "LLM-NPU Towards Efficient Foundation Model Inference on Low-Power NPUs"),
    (28, "Tummalapalli2026","LLM Inference at the Edge Mobile NPU and GPU Performance Efficiency Trade-offs"),
    (29, "CloudEdge2025",  "Cloud to Edge Benchmarking LLM Inference On Hardware-Accelerated Single-Board Computers"),
    # ── Operator Fusion and Compiler Optimization ────────────────────────
    (30, "Niu2021",        "DNNFusion Accelerating Deep Neural Networks Execution with Advanced Operator Fusion"),
    (31, "Zhang2025",      "Unified Operator Fusion for Heterogeneous Hardware in ML Inference Frameworks"),
    (32, "Forge2026",      "Forge-UGC FX Optimization and Register-Graph Engine for Universal Graph Compiler"),
    (33, "Parallax2025",   "Parallax Runtime Parallelization for Operator Fallbacks in Heterogeneous Edge Systems"),
    # ── Quantization ─────────────────────────────────────────────────────
    (34, "Jacob2018",      "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"),
    # ── Roofline and Performance Modeling ────────────────────────────────
    (35, "RooflineBench2026","RooflineBench A Benchmarking Framework for On-Device LLMs via Roofline Analysis"),
    # ── GNN Benchmarking and Scalability ────────────────────────────────
    (36, "Shirzad2023",    "Exphormer Sparse Transformers for Graphs"),
    (37, "Tonshoff2023",   "Where Did the Gap Go Reassessing the Long-Range Graph Benchmark"),
    (38, "Thomas2023",     "Graph Neural Networks Designed for Different Graph Types A Survey"),
    (39, "Zeng2020",       "GraphSAINT Graph Sampling Based Inductive Learning Method"),
    (40, "Cong2020",       "Minimal Variance Sampling with Provable Guarantees for Fast Training of GNNs"),
    # ── Additional GNN Acceleration / Systems ────────────────────────────
    (41, "TC_GNN2023",     "TC-GNN Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs"),
    (42, "HGL2025",        "HGL Accelerating Heterogeneous GNN Training with Holistic Representation and Optimization"),
    (43, "TT_GNN2023",     "TT-GNN Efficient On-Chip Graph Neural Network Training via Embedding Reformation"),
    (44, "DynaGraph2022",  "DynaGraph Dynamic Graph Neural Networks at Scale"),
    (45, "IANUS2024",      "IANUS Integrated Accelerator based on NPU-PIM Unified Memory System"),
    (46, "FlashMem2026",   "FlashMem Supporting Modern DNN Workloads on Mobile with GPU Memory Hierarchy Optimizations"),
    # ── Eyeriss (reference accelerator) ──────────────────────────────────
    (47, "Chen2016",       "Eyeriss An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks"),
    (48, "Chen2019",       "Eyeriss v2 A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices"),
    # ── MLPerf / Edge AI Roadmap ────────────────────────────────────────
    (49, "MLPerf2019",     "MLPerf Inference Benchmark"),
    (50, "EdgeAI2022",     "Roadmap for Edge AI A Dagstuhl Perspective"),
    # ── Lightweight / Recommendation GNNs ───────────────────────────────
    (51, "He2020",         "LightGCN Simplifying and Powering Graph Convolution Network for Recommendation"),
    (52, "LightGNN2025",   "LightGNN Simple Graph Neural Network for Recommendation"),
    # ── EIE (Sparse Accelerator baseline) ───────────────────────────────
    (53, "Han2016",        "EIE Efficient Inference Engine on Compressed Deep Neural Network"),
    # ── Related GNN Works ────────────────────────────────────────────────
    (54, "Zhang2022graphless","Graph-less Neural Networks Teaching Old MLPs New Tricks Via Distillation"),
    (55, "Tailor2022",     "Do We Need Anisotropic Graph Neural Networks (EGC)"),
    (56, "Bayraktar2026",  "Beyond GNNs A Methodological Benchmark of Feature Efficiency for Link Prediction"),
    (57, "Singh2023",      "Edge AI A survey"),
]

# Map from the *current* filename prefix (number) to canonical entry
# We do a best-effort match by scanning existing .md files
CURRENT_TO_CANONICAL_HINT = {
    "2":  10,   # Deep Residual → He2016 ResNet
    "3 - EIE": 53,
    "3 - Published": 1,   # GCN — ICLR 2017
    "4":  8,    # MPNN
    "5":  23,   # TPU
    "6":  4,    # GraphSAGE
    "7":  2,    # GAT — ICLR 2018
    "8 - MobileNetV2": 11,
    "8 - arXiv1712": 34,  # Quantization
    "9":  7,    # APPNP — ICLR 2019
    "10": 7,    # also APPNP — duplicate numbering, both map to APPNP (ICLR 2019)
    "11": 6,    # SGC
    "12": 39,   # GraphSAINT ICLR 2020
    "13": 14,   # BERT-tiny (Well-Read Students)
    "14 - EnGN": 17,
    "14 - MLPerf": 49,
    "15": 18,   # HyGCN — HPCA 2020
    "16": 51,   # LightGCN
    "17": 15,   # OGB
    "18": 40,   # Minimal Variance Sampling
    "19": 19,   # GRIP
    "20": 5,    # GIN — ICLR 2021
    "21": 20,   # Computing GNNs survey
    "22": 9,    # Graph Transformer
    "24": 55,   # Anisotropic GNNs
    "25": 38,   # GNN Types Survey
    "26": 30,   # DNNFusion
    "27": 36,   # Exphormer
    "28": 50,   # Roadmap Edge AI
    "29": 41,   # TC-GNN
    "30": 38,   # GNN Types Survey (duplicate, same canonical)
    "31": 37,   # Where Did the Gap Go (arXiv 2210.04055)
    "32": 21,   # GNN Acceleration Survey
    "33": 37,   # Where Did the Gap Go
    "34": 26,   # Fast LLM NPU
    "35": 45,   # IANUS
    "36": 52,   # LightGNN (arXiv 2501.00636)
    "37": 52,   # LightGNN
    "38": 33,   # Parallax
    "39": 35,   # RooflineBench
    "40": 46,   # FlashMem
    "41": 28,   # LLM Inference Edge
    "42": 32,   # Forge-UGC
    "43": 29,   # Cloud to Edge
    "44": 24,   # Intel Core Ultra NPU
    "45": 16,   # Hardware Acceleration GNNs
    "46": 47,   # Eyeriss
    "47": 48,   # Eyeriss v2
    "48": 22,   # GNNMark
    "49": 44,   # DynaGraph
    "50": 42,   # HGL
    "51": 34,   # Quantization
    "52": 27,   # LLM-NPU
    "53": 15,   # OGB
    "54": 36,   # Exphormer
    "55": 43,   # TT-GNN
    "56": 31,   # Unified Operator Fusion
    "1-s2.0-S2667345223000196-main": 57,  # Edge AI Survey
}


def safe_filename(s: str, max_len: int = 80) -> str:
    """Strip characters unsafe for filenames, collapse spaces."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s


def canonical_name(entry) -> str:
    num, slug, title = entry
    clean_title = safe_filename(title)
    return f"{num:02d} - {slug} - {clean_title}.md"


def find_md_file(prefix: str) -> Path | None:
    """Find existing .md file matching a number prefix (e.g. '3 - EIE...')."""
    for f in REFS_DIR.glob("*.md"):
        if f.name.startswith(prefix):
            return f
    return None


def resolve_current_file(hint_key: str) -> Path | None:
    """Given a hint key like '3 - EIE', find the matching .md."""
    for f in REFS_DIR.glob("*.md"):
        if f.stem.startswith(hint_key):
            return f
        # also match bare number
        if f.stem.startswith(hint_key + " -") or f.stem.startswith(hint_key + "-"):
            return f
    # bare number match
    num = hint_key.split(" ")[0]
    for f in REFS_DIR.glob(f"{num} - *.md"):
        return f
    return None


# ---------------------------------------------------------------------------
# docs/references.md content — structured reference list
# ---------------------------------------------------------------------------

REFERENCES_MD = """\
# Reference Library

Comprehensive list of all papers and technical reports referenced in
*Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis*.
Each entry includes a brief annotation explaining its relevance to the benchmark.

> BibTeX source: [`paper/references.bib`](../paper/references.bib)  
> Full text (where available): `paper/references/` directory

---

## 1. GNN Architectures (Models Benchmarked)

These are the nine GNN architectures evaluated on the Intel Core Ultra NPU.

| # | Citation | Venue | Role in Benchmark |
|---|----------|-------|-------------------|
| [1] | Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, 2017 | ICLR 2017 | **GCN** — spectral convolution baseline; foundational GNN model |
| [2] | Veličković et al., *Graph Attention Networks*, 2018 | ICLR 2018 | **GAT** — attention-based GNN; fails INT8 NPU compilation |
| [3] | Brody et al., *How Attentive are Graph Attention Networks?*, 2022 | ICLR 2022 | **GATv2** — dynamic attention variant; also fails INT8 on NPU |
| [4] | Hamilton et al., *Inductive Representation Learning on Large Graphs*, 2017 | NeurIPS 2017 | **GraphSAGE** — sampling-based inductive learning |
| [5] | Xu et al., *How Powerful are Graph Neural Networks?*, 2019 | ICLR 2019 | **GIN** — maximum expressivity; isomorphism network |
| [6] | Wu et al., *Simplifying Graph Convolutional Networks*, 2019 | ICML 2019 | **SGC** — removes non-linearities; shows INT8 paradox on NPU |
| [7] | Gasteiger et al., *Predict then Propagate: GNNs meet Personalized PageRank*, 2019 | ICLR 2019 | **APPNP** — propagation decoupled from prediction |
| [8] | Gilmer et al., *Neural Message Passing for Quantum Chemistry*, 2017 | ICML 2017 | **MPNN** — general message-passing framework |
| [9] | Dwivedi & Bresson, *A Generalization of Transformer Networks to Graphs*, 2021 | arXiv | **GraphTransformer** — hybrid GNN-Transformer architecture |

---

## 2. Dense Baseline Models

Five dense vision/NLP models used to contrast NPU behavior on its intended workload.

| # | Citation | Venue | Role |
|---|----------|-------|------|
| [10] | He et al., *Deep Residual Learning for Image Recognition*, 2016 | CVPR 2016 | **ResNet-50** — dense NPU baseline (3.94 ms mean) |
| [11] | Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, 2018 | CVPR 2018 | **MobileNetV2** — best NPU latency (1.90 ms mean) |
| [12] | Tan & Le, *EfficientNet: Rethinking Model Scaling*, 2019 | ICML 2019 | **EfficientNet-B0** — INT8 compilation fails on NPU |
| [13] | Dosovitskiy et al., *An Image is Worth 16×16 Words: ViT*, 2021 | ICLR 2021 | **ViT-Tiny** — Vision Transformer; INT8 fails |
| [14] | Turc et al., *Well-Read Students Learn Better (BERT-Tiny)*, 2019 | arXiv | **BERT-Tiny** — NLP Transformer; demonstrates Fusion Overhead Paradox |

---

## 3. Datasets

| # | Citation | Dataset | Used For |
|---|----------|---------|---------|
| [15] | Hu et al., *Open Graph Benchmark*, 2020 | NeurIPS 2020 | **ogbn-arxiv**, **ogbn-proteins**, **ogbn-products** — all three datasets in benchmark |

---

## 4. GNN Hardware Acceleration

Key references on accelerator design and performance characterization for GNNs.

| # | Citation | Venue | Key Contribution |
|---|----------|-------|-----------------|
| [16] | Auten et al., *Hardware Acceleration of GNNs*, 2020 | DAC 2020 | First GNN-specific accelerator; 7.5× over GPU |
| [17] | Liang et al., *EnGN: High-Throughput GNN Accelerator*, 2021 | IEEE TC | Ring-edge-reduce dataflow; 1800× over CPU |
| [18] | Yan et al., *HyGCN: Hybrid GCN Accelerator*, 2020 | HPCA 2020 | Aggregation+combination hybrid; 1509× over CPU |
| [19] | Kiningham et al., *GRIP: GNN Accelerator Architecture*, 2023 | IEEE TC | Low-latency GNN inference; 17× over CPU |
| [20] | Abadal et al., *Computing GNNs: Algorithms to Accelerators*, 2021 | ACM CSUR | Comprehensive survey; framing memory-bound argument |
| [21] | Zhang et al., *GNN Acceleration Survey*, 2026 | ACM CSUR | Taxonomy of GNN acceleration techniques |
| [22] | Baruah et al., *GNNMark: Benchmark Suite for GNN Training*, 2021 | ISPASS 2021 | GPU-based GNN benchmark; comparison baseline |

---

## 5. Intel NPU and AI Hardware Architecture

| # | Citation | Source | Relevance |
|---|----------|--------|-----------|
| [23] | Jouppi et al., *In-Datacenter Performance Analysis of a TPU*, 2017 | ISCA 2017 | TPU as domain-specific accelerator reference |
| [24] | Intel Corp., *Heterogeneous AI Powerhouse: Intel Core Ultra NPU*, 2024 | Whitepaper | Official Meteor Lake NPU architecture (NPU 3720) |
| [25] | Lam, *Intel Meteor Lake's NPU*, 2024 | Chips & Cheese | Technical NPU microarchitecture analysis |
| [47] | Chen et al., *Eyeriss: Energy-Efficient CNN Accelerator*, 2016 | ISSCC 2016 | Reference NPU design baseline |
| [48] | Chen et al., *Eyeriss v2: Flexible DNN Accelerator*, 2019 | IEEE JETCAS | Mobile NPU design reference |

---

## 6. Edge AI and NPU Inference

| # | Citation | Venue | Key Finding |
|---|----------|-------|------------|
| [26] | Xu et al., *Fast On-device LLM Inference with NPUs*, 2025 | ASPLOS 2025 | NPU-specific operator optimization strategies |
| [27] | Gao et al., *LLM-NPU: Efficient Foundation Model Inference on NPUs*, 2025 | IEEE CS 2025 | Memory bandwidth as NPU bottleneck |
| [28] | Tummalapalli & Arayakandy, *LLM Inference at the Edge*, 2026 | arXiv 2026 | Mobile/NPU/GPU sustained load trade-offs |
| [29] | Kachris et al., *Cloud to Edge: Benchmarking LLM Inference*, 2025 | arXiv 2025 | Hardware-accelerated single-board benchmark |
| [45] | Heo et al., *IANUS: NPU-PIM Unified Memory System*, 2024 | ASPLOS 2024 | NPU-PIM co-design for memory-bound workloads |
| [57] | Singh & Gill, *Edge AI: A survey*, 2023 | IoT and Cyber-Physical Systems 2023 | Comprehensive survey of edge computing paradigms and transition to Edge AI |

---

## 7. Operator Fusion and Compiler Optimization

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [30] | Niu et al., *DNNFusion: Advanced Operator Fusion*, 2021 | PLDI 2021 | Operator fusion theory; motivates FGR metric |
| [31] | Zhang et al., *Unified Operator Fusion for Heterogeneous Hardware*, 2025 | arXiv 2025 | Cross-device fusion optimization |
| [32] | Zhang et al., *Forge-UGC: Universal Graph Compiler*, 2026 | arXiv 2026 | Graph-level compiler optimization |
| [33] | Tang et al., *Parallax: Adaptive DAG Partitioning for CPU Fallbacks*, 2025 | arXiv 2025 | CPU fallback scheduling — directly related to our fallback detection |
| [46] | Shu et al., *FlashMem: DNN Workloads on Mobile GPU*, 2026 | arXiv 2026 | Memory hierarchy optimization for mobile inference |
| [35] | Pagoda et al., *RooflineBench: On-device LLM Benchmarking*, 2026 | arXiv 2026 | Roofline methodology for edge accelerators |

---

## 8. Quantization

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [34] | Jacob et al., *Quantization and Training of NNs for Integer-Arithmetic Inference*, 2018 | CVPR 2018 | Foundation for integer-only inference; this repository uses ONNX Runtime dynamic quantization |

---

## 9. GNN Scalability and Benchmarking

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [36] | Shirzad et al., *Exphormer: Sparse Transformers for Graphs*, 2023 | ICML 2023 | Sparse graph transformer; comparison for graph density analysis |
| [37] | Tönshoff et al., *Where Did the Gap Go?*, 2023 | LoG 2023 | Rigorous GNN benchmarking methodology |
| [38] | Thomas et al., *GNNs Designed for Different Graph Types*, 2023 | TMLR 2023 | Survey of graph type diversity |
| [39] | Zeng et al., *GraphSAINT: Graph Sampling for Inductive Learning*, 2020 | ICLR 2020 | Sampling-based training scalability |
| [40] | Cong et al., *Minimal Variance Sampling for GNNs*, 2020 | KDD 2020 | Variance reduction in GNN training |
| [41] | Huang et al., *TC-GNN: Sparse GNN on Dense Tensor Cores*, 2023 | USENIX ATC 2023 | GPU sparse-dense bridge for GNN ops |
| [42] | Fan et al., *HGL: Heterogeneous GNN Training*, 2025 | IEEE 2025 | GNN training optimization |
| [43] | Qu et al., *TT-GNN: On-Chip GNN Training*, 2023 | MICRO 2023 | Tensor-train GNN; on-chip memory-efficient training |
| [44] | Guan et al., *DynaGraph: Dynamic GNNs at Scale*, 2022 | SIGMOD 2022 | Dynamic GNN optimization |
| [54] | Zhang et al., *Graph-less Neural Networks via Distillation*, 2022 | ICLR 2022 | MLP distillation vs GNN inference |
| [55] | Tailor et al., *Do We Need Anisotropic GNNs?*, 2022 | ICLR 2022 | Isotropic vs anisotropic GNN efficiency |

---

## 10. Lightweight and Recommendation GNNs

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [51] | He et al., *LightGCN*, 2020 | SIGIR 2020 | Simplified GCN for recommendation; NPU-friendly structure |
| [52] | Cai et al., *LightGNN*, 2025 | arXiv 2025 | Ultra-lightweight GNN variant |

---

## 11. Related Work and Context

| # | Citation | Venue | Relevance |
|---|----------|-------|-----------|
| [49] | Mattson et al., *MLPerf Inference Benchmark*, 2020 | IEEE Micro 2020 | Inference benchmarking standard |
| [50] | Dhar et al., *Roadmap for Edge AI: A Dagstuhl Perspective*, 2022 | Commun. ACM | Edge AI research roadmap |
| [53] | Han et al., *EIE: Efficient Inference Engine on Compressed DNNs*, 2016 | ISCA 2016 | Sparse inference engine; NPU comparison context |
| [56] | Bayraktar, *Beyond GNNs: Feature Efficiency for Link Prediction*, 2026 | KAIS 2026 | Challenges automatic GNN preference on sparse graphs |
"""


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    print(f"{'[DRY RUN] ' if dry_run else ''}Organizing {REFS_DIR}")

    # 1. Delete junk files
    print("\n── Step 1: Removing junk / personal files ──")
    for name in JUNK_FILES:
        path = REFS_DIR / name
        if path.exists():
            print(f"  DELETE  {name}")
            if not dry_run:
                path.unlink()
        else:
            # Try glob for partial matches
            matches = list(REFS_DIR.glob(f"*{name}*"))
            for m in matches:
                if m.name in JUNK_FILES or any(j in m.name for j in JUNK_FILES):
                    print(f"  DELETE  {m.name}")
                    if not dry_run:
                        m.unlink()

    # 2. Build canonical name map
    canonical_map = {entry[0]: entry for entry in CANONICAL}

    # 3. Rename files (both .md and .pdf)
    print("\n── Step 2: Renaming reference files (.md and .pdf) ──")
    existing_files = sorted(list(REFS_DIR.glob("*.md")) + list(REFS_DIR.glob("*.pdf")))

    renamed = {}
    used_targets = set()

    for f in existing_files:
        if f.name in {"index.md", "academic-sources-summary.md"}:
            continue

        ext = f.suffix.lower()[1:]  # 'md' or 'pdf'
        stem = f.stem

        # Determine which canonical entry this file maps to
        canonical_num = None

        # A. First try matching by canonical slug contained in the filename (e.g. "Kipf2017" or "Bayraktar2026")
        for entry in CANONICAL:
            num, slug, title = entry
            # Match slug as a word boundary
            if re.search(r"\b" + re.escape(slug) + r"\b", stem, re.IGNORECASE):
                canonical_num = num
                break

        # B. If no slug match, try prefix-based matching from CURRENT_TO_CANONICAL_HINT
        if canonical_num is None:
            sorted_hints = sorted(CURRENT_TO_CANONICAL_HINT.keys(), key=len, reverse=True)
            for hint in sorted_hints:
                if hint.isdigit():
                    if re.match(r"^" + hint + r"\b", stem):
                        canonical_num = CURRENT_TO_CANONICAL_HINT[hint]
                        break
                else:
                    if stem.startswith(hint):
                        canonical_num = CURRENT_TO_CANONICAL_HINT[hint]
                        break

        # C. Special case: if it contains "s10115" (Bayraktar DOI prefix), map to Bayraktar (56)
        if canonical_num is None:
            if "s10115" in stem.lower():
                canonical_num = 56

        if canonical_num and canonical_num in canonical_map:
            entry = canonical_map[canonical_num]
            num, slug, title = entry
            clean_title = safe_filename(title)
            new_name = f"{num:02d} - {slug} - {clean_title}.{ext}"

            # Handle duplicates (multiple old files → same canonical)
            target_key = (new_name, ext)
            if target_key in used_targets:
                # append suffix
                base = new_name[:-(len(ext)+1)]
                new_name = f"{base}_alt.{ext}"
            used_targets.add(target_key)

            new_path = REFS_DIR / new_name
            if f != new_path:
                print(f"  RENAME  {f.name!r}")
                print(f"       →  {new_name!r}")
                if not dry_run:
                    # If target already exists (e.g. from previous partial runs), remove it first to avoid collision
                    if new_path.exists():
                        new_path.unlink()
                    f.rename(new_path)
                renamed[f.name] = new_name
        else:
            print(f"  SKIP    {f.name!r}  (no canonical mapping found)")

    # 4. Write docs/references.md
    print("\n── Step 3: Writing docs/references.md ──")
    refs_md_path = DOCS_DIR / "references.md"
    if not dry_run:
        refs_md_path.write_text(REFERENCES_MD, encoding="utf-8")
    print(f"  WROTE   {refs_md_path}")

    # 5. Rewrite paper/references/index.md
    print("\n── Step 4: Rewriting paper/references/index.md ──")
    index_lines = ["# Reference Index\n\n"]
    index_lines.append(
        "> This index lists all academic reference files in this directory, "
        "organized by category.  \n"
        "> BibTeX source: [`paper/references.bib`](../references.bib)  \n"
        "> Human-readable summary: [`docs/references.md`](../../docs/references.md)\n\n"
    )
    index_lines.append("| # | File | Canonical Citation |\n")
    index_lines.append("|---|------|--------------------|\n")

    for num, slug, title in CANONICAL:
        fname = f"{num:02d} - {slug} - {safe_filename(title, 80)}.md"
        index_lines.append(f"| {num} | [{fname}]({fname}) | {slug}: *{title[:60]}{'...' if len(title)>60 else ''}* |\n")

    if not dry_run:
        (REFS_DIR / "index.md").write_text("".join(index_lines), encoding="utf-8")
    print(f"  WROTE   {REFS_DIR / 'index.md'}")

    print("\n✅ Done.")
    if dry_run:
        print("   (Dry run — no files were modified)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
