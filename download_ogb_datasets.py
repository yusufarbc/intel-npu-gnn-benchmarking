#!/usr/bin/env python3
"""
Download OGB (Open Graph Benchmark) datasets for GNN benchmarking.
Downloads: ogbn-arxiv, ogbn-products
Note: Reddit dataset (ogbn-reddit) is deprecated in OGB
"""

import sys
import os
import builtins
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Monkey-patch input to auto-confirm downloads
original_input = builtins.input
def auto_confirm_input(prompt):
    if "Will you proceed?" in prompt:
        print(f"{prompt}y")
        return "y"
    return original_input(prompt)
builtins.input = auto_confirm_input

def download_dataset(dataset_name: str, root: Path, dataset_type: str = "nodeproppred") -> None:
    """Download a single OGB dataset."""
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name} ({dataset_type})...")
    print(f"{'='*60}")
    
    try:
        if dataset_type == "nodeproppred":
            from ogb.nodeproppred import NodePropPredDataset
            dataset = NodePropPredDataset(name=dataset_name, root=str(root))
        elif dataset_type == "linkpred":
            from ogb.linkproppred import LinkPropPredDataset
            dataset = LinkPropPredDataset(name=dataset_name, root=str(root))
        elif dataset_type == "graphpred":
            from ogb.graphproppred import GraphPropPredDataset
            dataset = GraphPropPredDataset(name=dataset_name, root=str(root))
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    except ImportError as e:
        print(f"ERROR: ogb library not found. Install with: pip install ogb")
        sys.exit(1)
    
    try:
        print(f"[OK] Successfully downloaded {dataset_name}")
        print(f"   Dataset path: {dataset.root}")
        
        # Print dataset info if available
        if dataset_type == "nodeproppred":
            split_idx = dataset.get_idx_split()
            print(f"   Number of nodes: {len(split_idx['train']) + len(split_idx['valid']) + len(split_idx['test'])}")
        elif dataset_type == "linkpred":
            split_idx = dataset.get_edge_split()
            print(f"   Graph downloaded successfully")
        elif dataset_type == "graphpred":
            split_idx = dataset.get_idx_split()
            print(f"   Number of graphs: {len(split_idx['train']) + len(split_idx['valid']) + len(split_idx['test'])}")
        
    except Exception as e:
        print(f"[ERROR] Failed to download {dataset_name}: {e}")
        raise

def main():
    """Download all required OGB datasets."""
    # Set data directory
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    
    # Datasets to download with their types
    # Note: ogbn-arxiv is already downloaded
    # Reddit dataset (ogbn-reddit) was deprecated in OGB and is no longer available
    datasets = [
        ("ogbn-proteins", "nodeproppred"),
    ]
    
    print("\nStarting dataset downloads...")
    print("This may take several minutes depending on your internet connection.")
    
    for dataset_name, dataset_type in datasets:
        try:
            download_dataset(dataset_name, data_dir, dataset_type)
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Dataset download process complete!")
    print("="*60)
    
    # List downloaded datasets
    print("\nDownloaded datasets:")
    for item in data_dir.iterdir():
        if item.is_dir():
            print(f"  - {item.name}")

if __name__ == "__main__":
    main()
