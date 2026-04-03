from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd


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
    iterations: int = 100
    warmup_iterations: int = 5
    random_seed: int = 42


class ProviderSelector:
    """Select execution providers with NPU-first fallback ordering."""

    PREFERRED_ORDER = [
        "OpenVINOExecutionProvider",
        "QNNExecutionProvider",
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

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
    def save_bar_chart(dataframe: pd.DataFrame, output_path: Path) -> None:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(dataframe["mode"], dataframe["avg_latency_ms"], color=["#8fb339", "#26547c"])
        plt.ylabel("Average Latency (ms)")
        plt.xlabel("Execution Mode")
        plt.title("ONNX Runtime: Fusion Disabled vs Enabled")

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


class NPUBenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.selected_providers = ProviderSelector.select()

    def _create_session(self, mode: BenchmarkMode) -> ort.InferenceSession:
        session_options = ort.SessionOptions()
        session_options.enable_profiling = True
        session_options.graph_optimization_level = mode.graph_level

        return ort.InferenceSession(
            str(self.config.model_path),
            sess_options=session_options,
            providers=self.selected_providers,
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
                values = rng.integers(0, 10, size=input_shape, dtype=np_dtype)
            else:
                values = np.zeros(input_shape, dtype=np_dtype)

            prepared_inputs[input_meta.name] = values

        return prepared_inputs

    @staticmethod
    def _resolve_dim(dim: object) -> int:
        if isinstance(dim, int) and dim > 0:
            return dim
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
        profiling_data_fn = getattr(session, "get_profiling_data", None)

        if callable(profiling_data_fn):
            profiling_data = profiling_data_fn()
            with output_path.open("w", encoding="utf-8") as file:
                if isinstance(profiling_data, (dict, list)):
                    json.dump(profiling_data, file, indent=2)
                elif isinstance(profiling_data, str):
                    try:
                        json.dump(json.loads(profiling_data), file, indent=2)
                    except json.JSONDecodeError:
                        file.write(profiling_data)
                else:
                    file.write(str(profiling_data))
            return output_path

        profiling_trace_file = Path(session.end_profiling())
        shutil.copyfile(profiling_trace_file, output_path)
        return output_path

    def _run_mode(self, mode: BenchmarkMode) -> Tuple[float, float, Path, List[str]]:
        session = self._create_session(mode)
        inputs = self._prepare_inputs(session)

        for _ in range(self.config.warmup_iterations):
            session.run(None, inputs)

        latencies_ms: List[float] = []

        # DAG node merging (operator fusion) keeps asymptotic compute at O(N + E)
        # while potentially reducing memory traffic and launch overhead constants.
        for _ in range(self.config.iterations):
            start = time.perf_counter()
            session.run(None, inputs)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

        # In memory-wall analysis, effective I/O pressure can dominate despite same O(N) arithmetic work.
        profiling_path = self._save_profiling(session, mode)

        mean_latency = float(np.mean(latencies_ms))
        std_latency = float(np.std(latencies_ms))
        return mean_latency, std_latency, profiling_path, session.get_providers()

    def run(self) -> pd.DataFrame:
        records = []

        for mode in [BenchmarkMode.BASELINE, BenchmarkMode.OPTIMIZED]:
            mean_latency, std_latency, profile_path, active_providers = self._run_mode(mode)
            records.append(
                {
                    "mode": mode.slug,
                    "avg_latency_ms": mean_latency,
                    "std_latency_ms": std_latency,
                    "iterations": self.config.iterations,
                    "providers": " | ".join(active_providers),
                    "profiling_json": str(profile_path),
                }
            )

        results_df = pd.DataFrame.from_records(records)
        csv_path = self.config.results_dir / "performance_summary.csv"
        results_df.to_csv(csv_path, index=False)

        chart_path = self.config.results_dir / "performance_comparison.png"
        ResultsVisualizer.save_bar_chart(results_df, chart_path)

        return results_df


def parse_args() -> BenchmarkConfig:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Benchmark ONNX graph optimization effects for NPU/CPU execution providers."
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model file.")
    parser.add_argument("--iterations", type=int, default=100, help="Measured inference iterations.")
    parser.add_argument("--results-dir", default=str(project_root / "results"), help="Output directory.")

    args = parser.parse_args()

    return BenchmarkConfig(
        model_path=Path(args.model).resolve(),
        results_dir=Path(args.results_dir).resolve(),
        iterations=args.iterations,
    )


def main() -> None:
    config = parse_args()
    ONNXModelValidator.validate(config.model_path)

    runner = NPUBenchmarkRunner(config)
    dataframe = runner.run()

    print("Benchmark complete.")
    print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
