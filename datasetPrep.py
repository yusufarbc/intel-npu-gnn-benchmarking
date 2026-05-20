#!/usr/bin/env python3
"""
Download required OGB datasets for benchmarking.
Downloads: ogbn-arxiv, ogbn-proteins, ogbn-products
"""

import builtins
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Monkey-patch input to auto-confirm downloads
_original_input = builtins.input

def _auto_confirm_input(prompt: str):
    if "Will you proceed?" in prompt:
        print(f"{prompt}y")
        return "y"
    return _original_input(prompt)

builtins.input = _auto_confirm_input


def _maybe_patch_torch_load() -> None:
    """Optionally allow legacy torch.load for OGB datasets (trusted files only)."""
    allow = os.environ.get("OGB_ALLOW_UNSAFE_TORCH_LOAD", "").strip().lower()
    if allow in {"1", "true", "yes"}:
        try:
            import torch
        except Exception as exc:
            print(f"WARNING: Could not import torch to patch torch.load: {exc}")
            return

        _orig_load = torch.load

        def _patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = _patched_load
        print(
            "WARNING: Using torch.load(weights_only=False) for OGB dataset files. "
            "Only enable this if you trust the download source."
        )
    else:
        print(
            "NOTE: If downloads fail with a weights_only error, set "
            "OGB_ALLOW_UNSAFE_TORCH_LOAD=1 and re-run."
        )


def _download_dataset(dataset_name: str, root: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"Downloading {dataset_name} (nodeproppred)...")
    print(f"{'=' * 60}")
    _maybe_patch_torch_load()
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError:
        print("ERROR: ogb library not found. Install with: pip install ogb")
        sys.exit(1)

    try:
        dataset = NodePropPredDataset(name=dataset_name, root=str(root))
        print(f"[OK] Successfully downloaded {dataset_name}")
        print(f"   Dataset path: {dataset.root}")
        split_idx = dataset.get_idx_split()
        total_nodes = len(split_idx["train"]) + len(split_idx["valid"]) + len(split_idx["test"])
        print(f"   Number of nodes: {total_nodes}")
    except Exception as exc:
        print(f"[ERROR] Failed to download {dataset_name}: {exc}")
        raise


def main() -> None:
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print("\nStarting dataset downloads...")
    print("This may take several minutes depending on your internet connection.")

    datasets = [
        "ogbn-arxiv",
        "ogbn-proteins",
        "ogbn-products",
    ]

    for name in datasets:
        try:
            _download_dataset(name, data_dir)
        except Exception:
            continue

    print("\n" + "=" * 60)
    print("Dataset download process complete!")
    print("=" * 60)

    print("\nDownloaded datasets:")
    for item in data_dir.iterdir():
        if item.is_dir():
            print(f"  - {item.name}")


if __name__ == "__main__":
    main()
