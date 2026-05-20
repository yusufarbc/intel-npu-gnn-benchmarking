import nbformat as nbf
from pathlib import Path

# Create new notebook with correct cell types
new_cells = []

# Cell 0: Title (markdown)
new_cells.append(nbf.v4.new_markdown_cell("""# Intel Core Ultra NPU: GNN Benchmarking Suite

Research pipeline for evaluating Graph Neural Networks (GNNs) on Intel Core Ultra NPU.
Run cells sequentially. Each visualization cell produces PNG + SVG outputs.

## Notebook Structure
1. **Setup** - Environment initialization
2. **Phase 1** - Model generation
3. **Phase 2** - Benchmarking (3 datasets, 100+ iterations)
4. **Phase 3** - Data analysis (4 analysis modules)
5. **Phase 4** - Visualizations (6 figures)
6. **Summary** - Output inventory"""))

# Cell 1: Environment Setup (code)
new_cells.append(nbf.v4.new_code_cell("""import os
import sys
import ctypes
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
print(f"Admin: {'Yes' if is_admin else 'No'}")

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("results/figures")

for d in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.style.use('default')
sns.set_theme(style="whitegrid")
print("Environment ready.")"""))

# Cell 2: Phase 1 - Model Generation (markdown)
new_cells.append(nbf.v4.new_markdown_cell("""## Phase 1: Model Generation

Generate GNN and baseline models:

**GNN Models:**
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE (Inductive representation learning)
- GIN (Graph Isomorphism Network)
- APPNP (Approximate Personalized Propagation)
- GraphTransformer (Transformer-based GNN)
- MPNN (Message Passing Neural Network)
- SGC (Simplified Graph Convolution)

**Baseline Models:**
- ResNet50 (CNN)
- MobileNetV2 (CNN)
- BERT-tiny (NLP/Transformer)"""))

# Cell 3: Run model prep (code)
new_cells.append(nbf.v4.new_code_cell("""!{sys.executable} analysis/model_prep.py"""))

# Cell 4: Phase 2 - Benchmarking (markdown)
new_cells.append(nbf.v4.new_markdown_cell("""## Phase 2: Benchmarking

Three benchmark configurations:

1. **Main Scalability** - Single model across iterations
   - 100 iterations, 3 repeats, 5 warmup
   
2. **Density Sweep** - Same models on 3 datasets
   - ogbn-arxiv (sparse, ~13 edges/node)
   - reddit (medium, ~50 edges/node)
   - ogbn-products (dense, ~25 edges/node)
   
3. **Scaling Analysis** - Varying graph sizes
   - Node counts: 512, 1024, 2048, 4096, 8192"""))

# Cell 5: Main benchmark (code)
new_cells.append(nbf.v4.new_code_cell("""# Main scalability benchmark with profiling
!{sys.executable} analysis/scalability_analyzer.py --iterations 100 --repeats 3 --profile --input-source auto --dataset-root data"""))

# Cell 6: Density sweep (code)
new_cells.append(nbf.v4.new_code_cell("""# Density sweep across 3 datasets with CPU+NPU comparison
!{sys.executable} analysis/density_sweep.py --models-dir models --results-dir results/density_sweep --datasets ogbn-arxiv,reddit,ogbn-products --iterations 100 --repeats 3 --profile --dataset-root data --auto-models --gnn-nodes 4096 --devices CPU,NPU"""))

# Cell 7: Scaling analysis (code)
new_cells.append(nbf.v4.new_code_cell("""# Scaling analysis with varying node counts
!{sys.executable} analysis/scaling_sweep.py --dataset ogbn-arxiv --device NPU --dataset-root data --sizes 512,1024,2048,4096,8192 --model GCN --out-dir results/scaling_sweep --iterations 100 --repeats 3 --warmup 5"""))

# Cell 8: Phase 3 - Data Analysis (markdown)
new_cells.append(nbf.v4.new_markdown_cell("""## Phase 3: Data Analysis

Process benchmark results into analysis tables.

Four analysis modules:
1. **Density Aggregation** - Merge sweep results across datasets
2. **Graph Topology** - Degree distribution and statistics
3. **Operator Composition** - ONNX operator breakdown
4. **CPU Fallback Patterns** - NPU vs CPU execution analysis"""))

# Cell 9: Density aggregation (code)
new_cells.append(nbf.v4.new_code_cell("""# Analysis 1: Density Aggregation
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.plot_config import apply_ieee_style

apply_ieee_style()

sweep_dir = Path('results/density_sweep')
out_dir = Path('results/figures')
out_dir.mkdir(parents=True, exist_ok=True)

def read_metadata(model_dir: Path):
    for p in sorted(model_dir.glob('run_*/input_metadata.json')):
        try:
            payload = json.loads(p.read_text())
            return {k: float(payload.get(k)) for k in ('used_num_nodes', 'used_num_edges') if isinstance(payload.get(k), (int, float))}
        except:
            continue
    return {}

rows = []
for ds_dir in sorted(sweep_dir.glob('dataset_*')):
    if not ds_dir.is_dir():
        continue
    matrix = ds_dir / 'scalability_matrix.csv'
    if not matrix.exists():
        continue
    df = pd.read_csv(matrix)
    df['dataset'] = ds_dir.name.replace('dataset_', '')
    meta = [read_metadata(ds_dir / m) for m in df['model'].astype(str)]
    df = pd.concat([df, pd.DataFrame(meta)], axis=1)
    if 'used_num_nodes' in df.columns and 'used_num_edges' in df.columns:
        df['edges_per_node'] = df['used_num_edges'] / df['used_num_nodes'].replace(0, np.nan)
    rows.append(df)

if rows:
    density_df = pd.concat(rows, ignore_index=True)
    density_df.to_csv(out_dir / 'density_analysis.csv', index=False)
    summary = density_df.groupby('dataset').agg(
        edges_per_node=('edges_per_node', 'mean'),
        latency_ms=('o_mean_ms', 'mean')
    ).sort_values('edges_per_node')
    print(f"Density data: {len(density_df)} records")
    print(summary)
else:
    print("No density data found")"""))

# Cell 10: Graph topology (code)
new_cells.append(nbf.v4.new_code_cell("""# Analysis 2: Graph Topology Statistics
from analysis.graph_topology_analyzer import GraphTopologyAnalyzer

analyzer = GraphTopologyAnalyzer(results_dir=out_dir, dataset_root=Path('data'))
stats_df = analyzer.analyze_datasets(['ogbn-arxiv', 'reddit', 'ogbn-products'])
print("\\nDataset Statistics:")
print(stats_df[['dataset_name', 'num_nodes', 'num_edges', 'avg_degree', 'density', 'power_law_alpha']].to_string(index=False))"""))

# Cell 11: Operator composition (code)
new_cells.append(nbf.v4.new_code_cell("""# Analysis 3: Operator Composition
import onnx
from analysis.plot_config import get_model_category

categories = ['SpMM/MatMul', 'MLP', 'Activation', 'Attention', 'Memory/Shape', 'Other']

def categorize(op_type):
    op = str(op_type)
    if op in {'MatMul', 'Gemm'}:
        return 'SpMM/MatMul'
    if op in {'Conv', 'ConvTranspose'}:
        return 'MLP'
    if 'Attention' in op or op in {'Softmax', 'LayerNormalization'}:
        return 'Attention'
    if op in {'Relu', 'Sigmoid', 'Tanh', 'Gelu'}:
        return 'Activation'
    if op in {'Gather', 'Scatter', 'Slice', 'Concat', 'Transpose', 'Reshape'}:
        return 'Memory/Shape'
    return 'Other'

rows = []
for model_path in sorted(Path('models').glob('*_fp32.onnx')):
    try:
        model = onnx.load(str(model_path))
        counts = {c: 0 for c in categories}
        for node in model.graph.node:
            counts[categorize(node.op_type)] += 1
        total = sum(counts.values()) or 1
        row = {k: (v/total)*100 for k, v in counts.items()}
        row['model'] = model_path.stem
        row['category'] = get_model_category(model_path.stem)
        rows.append(row)
    except Exception as e:
        print(f"Error analyzing {model_path}: {e}")

if rows:
    op_df = pd.DataFrame(rows).sort_values(['category', 'model'])
    op_df.to_csv(out_dir / 'operator_analysis.csv', index=False)
    print(f"\\nOperator analysis: {len(op_df)} models")
    print(op_df[['model', 'category', 'SpMM/MatMul', 'MLP', 'Attention']].to_string(index=False))
else:
    print("No operator data")"""))

# Cell 12: CPU fallback (code)
new_cells.append(nbf.v4.new_code_cell("""# Analysis 4: CPU Fallback Patterns
from analysis.ort_profile_utils import iter_operator_events, load_events

results_dir = Path('results')
model_fallback = {}

for model_dir in [p for p in results_dir.iterdir() if p.is_dir()]:
    traces = list(model_dir.glob('run_*/optimized_profiling.json'))
    if not traces:
        continue
    try:
        events = load_events(traces[-1])
        cpu_time = 0
        total_time = 0
        for _, dur, provider, _ in iter_operator_events(events):
            total_time += dur
            if 'cpu' in str(provider).lower():
                cpu_time += dur
        if total_time > 0:
            model_fallback[model_dir.name] = (cpu_time / total_time) * 100
    except:
        pass

if model_fallback:
    fallback_df = pd.DataFrame(list(model_fallback.items()), columns=['model', 'cpu_fallback_pct'])
    fallback_df.to_csv(out_dir / 'cpu_fallback_analysis.csv', index=False)
    print(f"\\nCPU fallback data: {len(fallback_df)} models")
    print(fallback_df.sort_values('cpu_fallback_pct', ascending=False).head(10).to_string(index=False))
else:
    print("No profiling data found")"""))

# Cell 13: Phase 4 - Visualizations (markdown)
new_cells.append(nbf.v4.new_markdown_cell("""## Phase 4: Visualizations

Six publication-ready figures. Each cell generates PNG (300 DPI) + SVG outputs."""))

# Cell 14: Figure 1 (code)
new_cells.append(nbf.v4.new_code_cell("""# Figure 1: Density vs Performance
# Shows relationship between graph density and NPU latency
import numpy as np
from analysis.plot_config import apply_ieee_style, savefig_ieee, IEEE_COLORS, SINGLE_COL

apply_ieee_style()
fig, ax = plt.subplots(figsize=SINGLE_COL)

if 'density_df' in locals() and not density_df.empty:
    npu_data = density_df[density_df.get('device', 'NPU') == 'NPU']
    for i, (dataset, group) in enumerate(npu_data.groupby('dataset')):
        if group.get('edges_per_node').notna().any():
            x = group['edges_per_node'].mean()
            y = group['o_mean_ms'].mean()
            ax.scatter(x, y, s=80, color=IEEE_COLORS[i % len(IEEE_COLORS)], 
                      label=dataset, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Average Degree (edges/node)', fontsize=9)
    ax.set_ylabel('Latency (ms)', fontsize=9)
    ax.set_title('Figure 1: Density vs Performance', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    savefig_ieee(fig, out_dir / 'fig1_density_vs_performance')
    display(Image(filename=out_dir / 'fig1_density_vs_performance.png'))
else:
    print("No density data available")
plt.close(fig)"""))

# Cell 15: Figure 2 (code)
new_cells.append(nbf.v4.new_code_cell("""# Figure 2: Degree Distribution (Log-Log)
# Shows power-law behavior of graph degree distributions
fig_path = out_dir / 'fig2_degree_distribution_loglog.png'
if fig_path.exists():
    display(Image(filename=fig_path))
    print(f"\\nFigure saved to: {fig_path}")
else:
    print("Degree distribution figure not found. Run graph topology analyzer first.")"""))

# Cell 16: Figure 3 (code)
new_cells.append(nbf.v4.new_code_cell("""# Figure 3: Operator Breakdown (Stacked Bars)
# Compares operator composition between GNN, CNN, and Transformer models
if 'op_df' in locals() and not op_df.empty:
    from analysis.plot_config import DOUBLE_COL, shorten_label
    
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    x = np.arange(len(op_df))
    bottom = np.zeros(len(op_df))
    colors = {cat: IEEE_COLORS[i % len(IEEE_COLORS)] for i, cat in enumerate(categories)}
    
    for cat in categories:
        vals = op_df[cat].fillna(0).values
        ax.bar(x, vals, bottom=bottom, label=cat, color=colors[cat], edgecolor='k', linewidth=0.25)
        bottom += vals
    
    # Category separators
    last_cat = None
    for i, cat in enumerate(op_df['category']):
        if cat != last_cat and i > 0:
            ax.axvline(i - 0.5, color='gray', lw=0.5, ls='--')
        last_cat = cat
    
    ax.set_xticks(x)
    ax.set_xticklabels([shorten_label(s, 12) for s in op_df['model']], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Share of ONNX nodes (%)', fontsize=9)
    ax.set_title('Figure 3: Operator Breakdown', fontsize=10)
    ax.legend(ncol=3, fontsize=7, loc='upper right')
    ax.set_ylim(0, 120)
    
    savefig_ieee(fig, out_dir / 'fig3_operator_breakdown')
    display(Image(filename=out_dir / 'fig3_operator_breakdown.png'))
    plt.close(fig)
else:
    print("No operator data available")"""))

# Cell 17: Figure 4 (code)
new_cells.append(nbf.v4.new_code_cell("""# Figure 4: CPU Fallback Heatmap
# Shows which models/operators fall back to CPU execution
if 'fallback_df' in locals() and not fallback_df.empty:
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    
    models = fallback_df['model'].tolist()
    values = fallback_df['cpu_fallback_pct'].tolist()
    
    colors = ['green' if v < 10 else 'orange' if v < 50 else 'red' for v in values]
    ax.barh(models, values, color=colors, edgecolor='black', linewidth=0.3)
    
    ax.set_xlabel('CPU Fallback (%)', fontsize=9)
    ax.set_ylabel('Model', fontsize=9)
    ax.set_title('Figure 4: CPU Fallback by Model', fontsize=10)
    ax.axvline(10, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='Low (<10%)')
    ax.axvline(50, color='orange', linestyle='--', linewidth=0.8, alpha=0.5, label='Medium (<50%)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='x')
    
    savefig_ieee(fig, out_dir / 'fig4_cpu_fallback')
    display(Image(filename=out_dir / 'fig4_cpu_fallback.png'))
    plt.close(fig)
else:
    print("No CPU fallback data available")"""))

# Cell 18: Figure 5 (code)
new_cells.append(nbf.v4.new_code_cell("""# Figure 5: Fusion Gain vs Performance
# Correlation between operator fusion and latency improvement
from scipy import stats

matrix_csv = Path('results/scalability_matrix.csv')

if matrix_csv.exists():
    df = pd.read_csv(matrix_csv)
    df['lat_impr_pct'] = (df['b_mean_ms'] - df['o_mean_ms']) / df['b_mean_ms'].replace(0, np.nan) * 100
    df['category'] = df['model'].apply(get_model_category)
    
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    markers = {'GNN (Irregular)': 'o', 'CNN (Regular)': 's', 'Transformer (Global Attn)': '^'}
    
    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat]
        ax.scatter(cat_data['speedup'], cat_data['lat_impr_pct'],
                  s=50, alpha=0.8, label=cat, marker=markers.get(cat, 'o'),
                  edgecolors='black', linewidth=0.5)
    
    ax.axvline(1.0, color='black', linestyle='--', linewidth=0.8)
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.6)
    
    valid = df[df['speedup'].notna() & df['lat_impr_pct'].notna()]
    if len(valid) > 3:
        r, p = stats.pearsonr(valid['speedup'], valid['lat_impr_pct'])
        ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    ax.set_xlabel('Fusion Gain Ratio', fontsize=9)
    ax.set_ylabel('Latency Improvement (%)', fontsize=9)
    ax.set_title('Figure 5: Fusion Gain vs Performance', fontsize=10)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    
    savefig_ieee(fig, out_dir / 'fig5_fusion_gain')
    display(Image(filename=out_dir / 'fig5_fusion_gain.png'))
    plt.close(fig)
else:
    print("Scalability matrix not found")"""))

# Cell 19: Figure 6 (code)
new_cells.append(nbf.v4.new_code_cell("""# Figure 6: Scaling Analysis
# Shows O(N) vs O(E) scaling behavior for GNN workloads
scaling_csv = Path('results/scaling_sweep/scaling_sweep.csv')

if scaling_csv.exists():
    df = pd.read_csv(scaling_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE_COL)
    
    ax1.plot(df['num_nodes'], df['o_mean_ms'], 'o-', color=IEEE_COLORS[0], 
            linewidth=1.5, markersize=6, label='Measured')
    ax1.set_xlabel('Number of Nodes', fontsize=9)
    ax1.set_ylabel('Latency (ms)', fontsize=9)
    ax1.set_title('Scaling by Nodes', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['num_edges'], df['o_mean_ms'], 's-', color=IEEE_COLORS[1],
            linewidth=1.5, markersize=6, label='Measured')
    ax2.set_xlabel('Number of Edges', fontsize=9)
    ax2.set_ylabel('Latency (ms)', fontsize=9)
    ax2.set_title('Scaling by Edges', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Figure 6: Scaling Characteristics', fontsize=10)
    plt.tight_layout()
    
    savefig_ieee(fig, out_dir / 'fig6_scaling')
    display(Image(filename=out_dir / 'fig6_scaling.png'))
    plt.close(fig)
    
    print(f"\\nScaling summary:")
    print(df[['num_nodes', 'num_edges', 'o_mean_ms']].to_string(index=False))
else:
    print("Scaling data not found")"""))

# Cell 20: Summary (markdown)
new_cells.append(nbf.v4.new_markdown_cell("""## Summary

Generated outputs in `results/figures/`:
- PNG (300 DPI) for publications
- SVG for vector editing
- CSV for data tables"""))

# Cell 21: Final code (code)
new_cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print("BENCHMARKING COMPLETE")
print("=" * 60)
print("\\nGenerated outputs in results/figures/:")

for ext in ['png', 'svg', 'csv']:
    files = sorted(out_dir.glob(f'*.{ext}'))
    if files:
        print(f"\\n{ext.upper()} files:")
        for f in files[:10]:  # Limit to 10 per type
            print(f"  - {f.name}")
        if len(files) > 10:
            print(f"  ... and {len(files)-10} more")

print("\\nAll figures exported as PNG (300 DPI) + SVG for publication")"""))

# Create new notebook
new_nb = nbf.v4.new_notebook()
new_nb['cells'] = new_cells

# Save
output_path = Path('npu_gnn_benchmarking_v2.ipynb')
nbf.write(new_nb, str(output_path))
print(f"Created: {output_path} with {len(new_cells)} cells")
