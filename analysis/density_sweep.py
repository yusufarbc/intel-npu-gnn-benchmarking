from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.scalability_analyzer import MultiModelPipeline, ScalabilityConfig




@dataclass
class SweepConfig:
    models_dir: Path
    results_dir: Path
    datasets: List[str]
    device: str = "NPU"  # backward-compat single-device
    devices: List[str] | None = None
    iterations: int = 100
    warmup_iterations: int = 5
    repeats: int = 3
    enable_profiling: bool = True
    dataset_root: Path | None = None
    auto_models: bool = False
    auto_models_int8: bool = False
    gnn_nodes: int = 4096
    gnn_features: int = 1433
    gnn_classes: int = 7
    auto_models_root: Path | None = None


def _ensure_dataset_gnn_models(
    *,
    dataset: str,
    models_root: Path,
    nodes: int,
    features: int,
    classes: int,
    quantize_int8: bool,
) -> Path:
    ds = dataset.strip().lower()
    edges_per_node = int(DEFAULT_EDGES_PER_NODE.get(ds, 25))
    edges = int(max(1, nodes * edges_per_node))
    out_dir = (models_root / "density_sweep_models" / f"{ds}_n{nodes}_e{edges}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # If directory already has ONNX files, reuse.
    if list(out_dir.glob("*.onnx")):
        return out_dir

    from analysis.model_prep import export_gnn_models

    print(f"[models] Exporting GNN suite for {ds}: nodes={nodes}, edges={edges} (≈{edges_per_node} edges/node)")
    export_gnn_models(
        out_dir,
        num_nodes=nodes,
        num_edges=edges,
        num_features=features,
        num_classes=classes,
        quantize_int8=quantize_int8,
    )
    return out_dir


def run_sweep(cfg: SweepConfig) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.auto_models:
        models_root = cfg.auto_models_root or cfg.models_dir
        models_root.mkdir(parents=True, exist_ok=True)
    else:
        models = sorted([m for m in cfg.models_dir.glob("*.onnx")])
        if not models:
            raise FileNotFoundError(f"No ONNX models found in: {cfg.models_dir}")

    devices = cfg.devices if cfg.devices else [cfg.device]

    for dev in devices:
        dev_slug = str(dev).strip()
        # Backward compatible layout when only one device requested.
        dev_root = cfg.results_dir if len(devices) == 1 else (cfg.results_dir / f"device_{dev_slug}")
        dev_root.mkdir(parents=True, exist_ok=True)

        for ds in cfg.datasets:
            ds_slug = ds.strip().lower()
            out_dir = dev_root / f"dataset_{ds_slug}"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Dataset sweep: {ds_slug} | device={dev_slug} ===")
            if cfg.auto_models:
                ds_models_dir = _ensure_dataset_gnn_models(
                    dataset=ds_slug,
                    models_root=models_root,
                    nodes=int(cfg.gnn_nodes),
                    features=int(cfg.gnn_features),
                    classes=int(cfg.gnn_classes),
                    quantize_int8=bool(cfg.auto_models_int8),
                )
                models = sorted([m for m in ds_models_dir.glob("*.onnx")])

            scfg = ScalabilityConfig(
                models=models,
                results_dir=out_dir,
                device=dev_slug,
                iterations=cfg.iterations,
                warmup_iterations=cfg.warmup_iterations,
                repeats=cfg.repeats,
                enable_profiling=cfg.enable_profiling,
                input_source=ds_slug,
                dataset_root=cfg.dataset_root,
            )
            MultiModelPipeline(scfg).run()

    print("\n[OK] Density sweep complete.")


def parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(description="Run scalability pipeline across multiple real graph datasets.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results/density_sweep")
    parser.add_argument(
        "--datasets",
        default="ogbn-arxiv,ogbn-proteins,ogbn-products",
        help="Comma-separated list: ogbn-arxiv, ogbn-proteins, ogbn-products (and/or cora).",
    )
    parser.add_argument("--device", default="NPU")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices to run (e.g., CPU,NPU). If set, overrides --device.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Enable ORT profiling (recommended).")
    parser.add_argument("--dataset-root", default=None, help="Root folder for dataset downloads/caches (default: ./data).")
    parser.add_argument(
        "--auto-models",
        action="store_true",
        help="Auto-export a GNN-only ONNX model suite per dataset with dataset-specific edge density.",
    )
    parser.add_argument(
        "--auto-models-int8",
        action="store_true",
        help="Enable INT8 quantization when auto-exporting per-dataset GNN models (slow, high RAM).",
    )
    parser.add_argument("--gnn-nodes", type=int, default=4096, help="Node count used when --auto-models is set")
    parser.add_argument("--gnn-features", type=int, default=1433, help="Feature dim used when --auto-models is set")
    parser.add_argument("--gnn-classes", type=int, default=7, help="Class count used when --auto-models is set")
    parser.add_argument(
        "--auto-models-root",
        default=None,
        help="Where to store auto-exported models (default: --models-dir)",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    devices = [d.strip() for d in str(args.devices).split(",") if d.strip()] if args.devices else None

    return SweepConfig(
        models_dir=Path(args.models_dir).resolve(),
        results_dir=Path(args.results_dir).resolve(),
        datasets=datasets,
        device=str(args.device),
        devices=devices,
        repeats=int(args.repeats),
        iterations=int(args.iterations),
        warmup_iterations=int(args.warmup),
        enable_profiling=bool(args.profile),
        dataset_root=Path(args.dataset_root).resolve() if args.dataset_root else None,
        auto_models=bool(args.auto_models),
        auto_models_int8=bool(args.auto_models_int8),
        gnn_nodes=int(args.gnn_nodes),
        gnn_features=int(args.gnn_features),
        gnn_classes=int(args.gnn_classes),
        auto_models_root=Path(args.auto_models_root).resolve() if args.auto_models_root else None,
    )


def main() -> None:
    cfg = parse_args()
    run_sweep(cfg)


if __name__ == "__main__":
    main()
