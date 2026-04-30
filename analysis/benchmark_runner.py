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
        plt.savefig(output_path, dpi=200)
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
        prepared_inputs: Dict[str, np.ndarray] = {}

        for input_meta in session.get_inputs():
            np_dtype = self._ort_type_to_numpy_dtype(input_meta.type)
            input_shape = [self._resolve_dim(dim) for dim in input_meta.shape]

            if np.issubdtype(np_dtype, np.floating):
                values = rng.random(input_shape, dtype=np.float32).astype(np_dtype)
            elif np.issubdtype(np_dtype, np.integer):
                # For GNN edge_index, we need valid node indices
                # Assuming input_meta.name is 'edge_index'
                if "edge_index" in input_meta.name.lower():
                    # We need to know num_nodes to generate valid indices
                    num_nodes = 2708 # Default Cora size
                    values = rng.integers(0, num_nodes, size=input_shape, dtype=np_dtype)
                else:
                    values = rng.integers(0, 10, size=input_shape, dtype=np_dtype)
            else:
                values = np.zeros(input_shape, dtype=np_dtype)

            prepared_inputs[input_meta.name] = values

        return prepared_inputs

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

    def _run_mode(self, mode: BenchmarkMode) -> Tuple[float, float, Path, List[str]]:
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
        mem_end_mb = mem_info.rss / (1024 * 1024)
        
        # In Windows, peak_wset is available. On Linux it's not. 
        # We'll use a generic approach for peak memory approximation.
        peak_mem_mb = getattr(mem_info, "peak_wset", mem_info.rss) / (1024 * 1024)

        mean_latency = float(np.mean(latencies_ms))
        std_latency = float(np.std(latencies_ms))
        
        return mean_latency, std_latency, cpu_end, peak_mem_mb, profiling_path, session.get_providers()

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

    args = parser.parse_args()

    return BenchmarkConfig(
        model_path=Path(args.model).resolve(),
        results_dir=Path(args.results_dir).resolve(),
        device=args.device,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        enable_profiling=args.profile,
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
