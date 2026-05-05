from __future__ import annotations

import argparse
import json
import os
os.environ["ORT_LOGGING_LEVEL"] = "4"
os.environ["OPENVINO_LOG_LEVEL"] = "0"
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import psutil


class BenchmarkMode(Enum):
    BASELINE = ("baseline", ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    OPTIMIZED = ("optimized", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)

    @property
    def slug(self) -> str:
        return self.value[0]

    @property
    def graph_level(self) -> ort.GraphOptimizationLevel:
        return self.value[1]


@dataclass
class BenchmarkConfig:
    model_path: Path
    results_dir: Path
    device: str = "NPU"
    iterations: int = 100
    warmup_iterations: int = 5
    random_seed: int = 42
    enable_profiling: bool = False
    input_source: str = "auto"  # auto|synthetic|cora|reddit|ogbn-arxiv|ogbn-products
    dataset_root: Path | None = None


class ProviderSelector:
    """Select execution providers with NPU-first fallback ordering."""

    PREFERRED_ORDER = [
        "OpenVINOExecutionProvider",
        "QNNExecutionProvider",
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    _dll_dir_cookies: List[object] = []

    @classmethod
    def prepare_runtime(cls) -> None:
        """Best-effort runtime prep for provider DLL loading."""
        if os.name != "nt":
            return

        add_dir = getattr(os, "add_dll_directory", None)
        if not callable(add_dir):
            return

        try:
            import openvino  # type: ignore

            base = Path(openvino.__file__).resolve().parent
            candidate_dirs = [base / "libs", base / "runtime" / "libs", base / "runtime", base]

            existing_path_parts = [part.strip('"') for part in os.environ.get("PATH", "").split(os.pathsep)]
            existing_lower = {part.lower() for part in existing_path_parts if part}

            for directory in candidate_dirs:
                if not directory.exists():
                    continue

                directory_str = str(directory)
                if directory_str.lower() not in existing_lower:
                    os.environ["PATH"] = directory_str + os.pathsep + os.environ.get("PATH", "")
                    existing_lower.add(directory_str.lower())

                try:
                    cookie = add_dir(directory_str)
                except OSError:
                    continue
                else:
                    cls._dll_dir_cookies.append(cookie)
        except Exception:
            pass

    @classmethod
    def select(cls) -> List[str]:
        available = ort.get_available_providers()
        selected = [provider for provider in cls.PREFERRED_ORDER if provider in available]

        if not selected and available:
            selected = available

        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")

        return selected


class ONNXModelValidator:
    @staticmethod
    def validate(model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        onnx.load(model_path)


class ResultsVisualizer:
    @staticmethod
    def save_bar_chart(dataframe: pd.DataFrame, output_path: Path, device_name: str) -> None:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(dataframe["mode"], dataframe["avg_latency_ms"], color=["#8fb339", "#26547c"])
        plt.ylabel("Average Latency (ms)")
        plt.xlabel("Execution Mode")
        plt.title(f"ONNX Runtime ({device_name}): Fusion Disabled vs Enabled")

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        ProviderSelector.prepare_runtime()
        self.selected_providers = ProviderSelector.select()

    def _create_session(self, mode: BenchmarkMode) -> ort.InferenceSession:
        session_options = ort.SessionOptions()
        session_options.enable_profiling = self.config.enable_profiling
        if self.config.enable_profiling:
            # By default ORT writes `onnxruntime_profile__*.json` into the process CWD.
            # Put them into the per-run results folder to avoid cluttering the project root.
            prefix = self.config.results_dir / f"{mode.slug}_onnxruntime_profile"
            session_options.profile_file_prefix = str(prefix)
        session_options.graph_optimization_level = mode.graph_level
        session_options.log_severity_level = 4

        providers_with_options: List[object] = []
        
        # Use OpenVINO for the target device if available, otherwise fallback.
        target_device = self.config.device.upper()
        
        for provider in self.selected_providers:
            if provider == "OpenVINOExecutionProvider":
                providers_with_options.append(
                    (
                        "OpenVINOExecutionProvider",
                        {
                            "device_type": target_device,
                        },
                    )
                )
            else:
                providers_with_options.append(provider)

        return ort.InferenceSession(
            str(self.config.model_path),
            sess_options=session_options,
            providers=providers_with_options,
        )

    def _prepare_inputs(self, session: ort.InferenceSession) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.config.random_seed)

        # --- detect graph-style models ---
        input_names = [i.name for i in session.get_inputs()]
        has_edge_index = any("edge_index" in name.lower() for name in input_names)
        has_x = any(name.lower() == "x" for name in input_names)

        resolved_source = (self.config.input_source or "auto").strip().lower()
        if resolved_source == "auto":
            resolved_source = "cora" if (has_edge_index and has_x) else "synthetic"

        dataset_payload: Dict[str, np.ndarray] = {}
        dataset_meta: Dict[str, object] = {"input_source": resolved_source}

        if resolved_source != "synthetic" and has_edge_index and has_x:
            dataset_payload, dataset_meta = self._prepare_graph_dataset_inputs(session, resolved_source)
            try:
                (self.config.results_dir / "input_metadata.json").write_text(
                    json.dumps(dataset_meta, indent=2), encoding="utf-8"
                )
            except Exception:
                pass

        prepared_inputs: Dict[str, np.ndarray] = {}
        for input_meta in session.get_inputs():
            if input_meta.name in dataset_payload:
                prepared_inputs[input_meta.name] = dataset_payload[input_meta.name]
                continue

            np_dtype = self._ort_type_to_numpy_dtype(input_meta.type)
            input_shape = [self._resolve_dim(dim) for dim in input_meta.shape]

            if np.issubdtype(np_dtype, np.floating):
                values = rng.random(input_shape, dtype=np.float32).astype(np_dtype)
            elif np.issubdtype(np_dtype, np.integer):
                if "edge_index" in input_meta.name.lower():
                    num_nodes = int(dataset_meta.get("used_num_nodes", 2708))
                    values = rng.integers(0, max(num_nodes, 1), size=input_shape, dtype=np_dtype)
                else:
                    values = rng.integers(0, 10, size=input_shape, dtype=np_dtype)
            else:
                values = np.zeros(input_shape, dtype=np_dtype)

            prepared_inputs[input_meta.name] = values

        return prepared_inputs

    def _prepare_graph_dataset_inputs(
        self, session: ort.InferenceSession, dataset_name: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        """Prepare `x` and `edge_index` from a real dataset topology.

        The repo's exported ONNX GNNs are mostly fixed-shape (Cora-like). To keep
        the toolchain stable, we sample/pad/truncate dataset tensors to match the
        ONNX input shapes.
        """
        from pathlib import Path as _Path

        import numpy as _np
        import torch as _torch

        try:
            from torch_geometric.data import Data  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "torch-geometric is required for real graph inputs. Install via requirements.txt."
            ) from exc

        data_root = self.config.dataset_root or (_Path(__file__).resolve().parent.parent / "data")
        _Path(data_root).mkdir(parents=True, exist_ok=True)

        name = dataset_name.strip().lower()
        data: Data
        if name in {"cora", "citeseer", "pubmed"}:
            from torch_geometric.datasets import Planetoid  # type: ignore

            dataset = Planetoid(root=str(_Path(data_root) / "planetoid"), name=name.capitalize())
            data = dataset[0]
        elif name == "reddit":
            from torch_geometric.datasets import Reddit  # type: ignore

            dataset = Reddit(root=str(_Path(data_root) / "reddit"))
            data = dataset[0]
        elif name in {"ogbn-arxiv", "ogbn-products"}:
            try:
                from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "OGBN datasets require the 'ogb' package. Install with: pip install ogb"
                ) from exc
            dataset = PygNodePropPredDataset(name=name, root=str(_Path(data_root) / "ogb"))
            data = dataset[0]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        input_metas = {i.name: i for i in session.get_inputs()}
        x_name = next((n for n in input_metas if n.lower() == "x"), None)
        edge_name = next((n for n in input_metas if "edge_index" in n.lower()), None)
        if x_name is None or edge_name is None:
            return {}, {"input_source": name, "note": "model_inputs_not_detected"}

        x_meta = input_metas[x_name]
        e_meta = input_metas[edge_name]
        x_shape = [self._resolve_dim(d) for d in x_meta.shape]
        e_shape = [self._resolve_dim(d) for d in e_meta.shape]

        expected_num_nodes = int(x_shape[0]) if len(x_shape) >= 1 else int(getattr(data, "num_nodes", 0) or 0)
        expected_num_features = int(x_shape[1]) if len(x_shape) >= 2 else 0
        expected_num_edges = int(e_shape[1]) if len(e_shape) >= 2 else int(getattr(data.edge_index, "shape", [2, 0])[1])

        rng = _np.random.default_rng(self.config.random_seed)

        x_tensor = getattr(data, "x", None)
        edge_index_t = data.edge_index.detach().cpu()

        dataset_num_nodes = int(getattr(data, "num_nodes", 0) or (int(x_tensor.shape[0]) if x_tensor is not None else 0) or 1)
        dataset_num_edges = int(edge_index_t.shape[1])
        dataset_num_features = int(x_tensor.shape[1]) if x_tensor is not None else int(expected_num_features or 1)

        def _tensor_x_for_subset(subset_nodes: _torch.Tensor) -> _np.ndarray:
            if x_tensor is None:
                # Synthesize features only for the selected subset.
                return rng.standard_normal((int(subset_nodes.numel()), int(expected_num_features or 1))).astype(_np.float32)
            return x_tensor[subset_nodes].detach().cpu().numpy().astype(_np.float32, copy=False)

        # --- Subgraph extraction (topology-faithful + avoids full NumPy edge copy) ---
        if expected_num_nodes > 0 and dataset_num_nodes > expected_num_nodes:
            try:
                from torch_geometric.utils import k_hop_subgraph, subgraph  # type: ignore

                seed = int(rng.integers(0, dataset_num_nodes))
                seed_t = _torch.tensor([seed], dtype=_torch.long)

                subset_nodes = None
                for hops in range(1, 6):
                    cand, _, _, _ = k_hop_subgraph(
                        seed_t,
                        num_hops=hops,
                        edge_index=edge_index_t,
                        num_nodes=dataset_num_nodes,
                        relabel_nodes=False,
                    )
                    if int(cand.numel()) >= int(expected_num_nodes):
                        subset_nodes = cand[: int(expected_num_nodes)]
                        break

                if subset_nodes is None:
                    subset_nodes = _torch.from_numpy(
                        rng.choice(dataset_num_nodes, size=int(expected_num_nodes), replace=False)
                    ).to(dtype=_torch.long)

                edge_index_sub, _ = subgraph(
                    subset_nodes,
                    edge_index_t,
                    relabel_nodes=True,
                    num_nodes=dataset_num_nodes,
                )
                x_np = _tensor_x_for_subset(subset_nodes)
                edge_index = edge_index_sub.detach().cpu().numpy().astype(_np.int64, copy=False)
                used_num_nodes = int(expected_num_nodes)
            except Exception:
                # Fallback: take first N nodes and filter edges.
                used_num_nodes = int(expected_num_nodes)
                subset_nodes = _torch.arange(used_num_nodes, dtype=_torch.long)
                x_np = _tensor_x_for_subset(subset_nodes)
                mask = (edge_index_t[0] < used_num_nodes) & (edge_index_t[1] < used_num_nodes)
                edge_index = edge_index_t[:, mask].detach().cpu().numpy().astype(_np.int64, copy=False)
        else:
            used_num_nodes = min(expected_num_nodes, dataset_num_nodes) if expected_num_nodes > 0 else dataset_num_nodes
            subset_nodes = _torch.arange(int(used_num_nodes), dtype=_torch.long)
            x_np = _tensor_x_for_subset(subset_nodes)
            mask = (edge_index_t[0] < used_num_nodes) & (edge_index_t[1] < used_num_nodes)
            edge_index = edge_index_t[:, mask].detach().cpu().numpy().astype(_np.int64, copy=False)

        used_num_features = expected_num_features if expected_num_features > 0 else dataset_num_features
        x_np = x_np[:, :used_num_features]
        if expected_num_nodes > 0 and x_np.shape[0] < expected_num_nodes:
            pad_nodes = expected_num_nodes - x_np.shape[0]
            x_np = _np.pad(x_np, ((0, pad_nodes), (0, 0)), mode="constant")
            used_num_nodes = int(expected_num_nodes)
        if expected_num_features > 0 and x_np.shape[1] < expected_num_features:
            pad_feats = expected_num_features - x_np.shape[1]
            x_np = _np.pad(x_np, ((0, 0), (0, pad_feats)), mode="constant")

        # Edge count adapt to expected E
        if expected_num_edges > 0 and edge_index.shape[1] > expected_num_edges:
            edge_index = edge_index[:, :expected_num_edges]
        elif expected_num_edges > 0 and edge_index.shape[1] < expected_num_edges:
            need = expected_num_edges - edge_index.shape[1]
            pad = rng.integers(0, max(used_num_nodes, 1), size=(2, need), dtype=_np.int64)
            edge_index = _np.concatenate([edge_index, pad], axis=1)

        x_np = x_np.astype(self._ort_type_to_numpy_dtype(x_meta.type), copy=False)
        edge_index = edge_index.astype(self._ort_type_to_numpy_dtype(e_meta.type), copy=False)

        meta: Dict[str, object] = {
            "input_source": name,
            "dataset_num_nodes": dataset_num_nodes,
            "dataset_num_edges": dataset_num_edges,
            "dataset_num_features": dataset_num_features,
            "expected_num_nodes": expected_num_nodes,
            "expected_num_edges": expected_num_edges,
            "expected_num_features": expected_num_features,
            "used_num_nodes": expected_num_nodes if expected_num_nodes > 0 else used_num_nodes,
            "used_num_edges": expected_num_edges if expected_num_edges > 0 else int(edge_index.shape[1]),
            "used_num_features": expected_num_features if expected_num_features > 0 else int(x_np.shape[1]),
            "note": "sampled_or_padded_to_match_onnx_shapes",
        }

        return {x_name: x_np, edge_name: edge_index}, meta

    @staticmethod
    def _resolve_dim(dim: object) -> int:
        if isinstance(dim, int) and dim > 0:
            return dim
        if isinstance(dim, str):
            if "num_nodes" in dim:
                return 2708
            if "num_edges" in dim:
                return 10000
        return 1

    @staticmethod
    def _ort_type_to_numpy_dtype(ort_type: str) -> np.dtype:
        type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }
        return type_map.get(ort_type, np.float32)

    def _save_profiling(self, session: ort.InferenceSession, mode: BenchmarkMode) -> Path:
        output_path = self.config.results_dir / f"{mode.slug}_profiling.json"
        
        try:
            profiling_trace_file = Path(session.end_profiling())
            shutil.copyfile(profiling_trace_file, output_path)
            # Remove the original profiling file to avoid clutter
            try:
                os.remove(profiling_trace_file)
            except Exception:
                pass
        except Exception:
            # Fallback for systems where end_profiling might fail
            pass
            
        return output_path

    def _run_mode(self, mode: BenchmarkMode) -> Tuple[float, float, float, float, Path, List[str]]:
        import gc
        session = self._create_session(mode)
        inputs = self._prepare_inputs(session)

        for _ in range(self.config.warmup_iterations):
            session.run(None, inputs)

        latencies_ms: List[float] = []
        process = psutil.Process(os.getpid())
        
        # Start tracking resource utilization
        cpu_start = process.cpu_percent(interval=None)
        mem_start_mb = process.memory_info().rss / (1024 * 1024)

        for _ in range(self.config.iterations):
            start = time.perf_counter()
            session.run(None, inputs)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

        profiling_path = self._save_profiling(session, mode)

        # End tracking resource utilization
        cpu_end = process.cpu_percent(interval=None)
        mem_info = process.memory_info()
        peak_mem_mb = getattr(mem_info, "peak_wset", mem_info.rss) / (1024 * 1024)

        mean_latency = float(np.mean(latencies_ms))
        std_latency = float(np.std(latencies_ms))
        
        # Explicitly destroy the session to free NPU/RAM resources
        del session
        gc.collect()
        
        return mean_latency, std_latency, cpu_end, peak_mem_mb, profiling_path, []

    def run(self) -> pd.DataFrame:
        records = []

        for mode in [BenchmarkMode.BASELINE, BenchmarkMode.OPTIMIZED]:
            mean_latency, std_latency, cpu_percent, peak_mem_mb, profile_path, active_providers = self._run_mode(mode)
            records.append(
                {
                    "mode": mode.slug,
                    "avg_latency_ms": mean_latency,
                    "std_latency_ms": std_latency,
                    "peak_memory_mb": peak_mem_mb,
                    "cpu_utilization_pct": cpu_percent,
                    "iterations": self.config.iterations,
                    "providers": " | ".join(active_providers),
                    "profiling_json": str(profile_path),
                }
            )

        results_df = pd.DataFrame.from_records(records)
        csv_path = self.config.results_dir / "performance_summary.csv"
        results_df.to_csv(csv_path, index=False)

        chart_path = self.config.results_dir / "performance_comparison.png"
        ResultsVisualizer.save_bar_chart(results_df, chart_path, self.config.device)

        return results_df


def parse_args() -> BenchmarkConfig:
    # Adjust project root because this script is now in engine/
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Benchmark ONNX graph optimization effects for specific devices."
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model file.")
    parser.add_argument("--device", default="NPU", choices=["CPU", "GPU", "NPU"], help="Target device (default: NPU).")
    parser.add_argument("--iterations", type=int, default=100, help="Measured inference iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--results-dir", default=str(project_root / "results"), help="Output directory.")
    parser.add_argument("--profile", action="store_true", help="Enable ONNX Runtime profiling traces.")
    parser.add_argument(
        "--input-source",
        default="auto",
        help="Input source: auto|synthetic|cora|reddit|ogbn-arxiv|ogbn-products (auto uses Cora for GNNs).",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root folder for dataset downloads/caches (default: ./data).",
    )

    args = parser.parse_args()

    return BenchmarkConfig(
        model_path=Path(args.model).resolve(),
        results_dir=Path(args.results_dir).resolve(),
        device=args.device,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        enable_profiling=args.profile,
        input_source=str(args.input_source),
        dataset_root=Path(args.dataset_root).resolve() if args.dataset_root else None,
    )


def main() -> None:
    config = parse_args()
    ONNXModelValidator.validate(config.model_path)

    runner = BenchmarkRunner(config)
    dataframe = runner.run()

    print(f"Benchmark on {config.device} complete.")
    print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
