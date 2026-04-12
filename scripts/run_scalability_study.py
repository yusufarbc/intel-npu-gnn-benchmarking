from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import shape_inference

from benchmark_npu import BenchmarkConfig, BenchmarkMode, NPUBenchmarkRunner


@dataclass
class ScalabilityConfig:
    models: List[Path]
    results_dir: Path
    iterations: int = 100
    warmup_iterations: int = 5
    repeats: int = 3
    peak_compute_gflops: float = 1000.0
    peak_bandwidth_gbps: float = 30.0


class ONNXGraphMetrics:
    @staticmethod
    def _dtype_nbytes(data_type: int) -> int:
        if data_type in {1, 6, 7, 11}:  # float32, int32, int64, float64
            return {1: 4, 6: 4, 7: 8, 11: 8}[data_type]
        if data_type in {10}:  # float16
            return 2
        if data_type in {2, 3, 4, 5, 9, 12, 13}:  # uint8/int8/uint16/int16/bool/uint32/uint64
            return {2: 1, 3: 1, 4: 2, 5: 2, 9: 1, 12: 4, 13: 8}[data_type]
        return 4

    @staticmethod
    def _shape_product(shape_dims: List[int]) -> int:
        product = 1
        for dim in shape_dims:
            if dim <= 0:
                return 0
            product *= dim
        return product

    @classmethod
    def _extract_shapes(cls, model: onnx.ModelProto) -> Dict[str, Tuple[List[int], int]]:
        shape_map: Dict[str, Tuple[List[int], int]] = {}

        def handle_value_info(value_info: Any) -> None:
            tensor_type = value_info.type.tensor_type
            if not tensor_type.HasField("shape"):
                return
            dims: List[int] = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    dims.append(int(dim.dim_value))
                else:
                    dims.append(1)
            shape_map[value_info.name] = (dims, int(tensor_type.elem_type) if tensor_type.elem_type else 1)

        for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            handle_value_info(vi)

        return shape_map

    @classmethod
    def _estimate_node_flops_bytes(
        cls,
        node: onnx.NodeProto,
        shape_map: Dict[str, Tuple[List[int], int]],
        initializer_map: Dict[str, onnx.TensorProto],
    ) -> Tuple[float, float]:
        op_type = node.op_type

        def tensor_elements(name: str) -> int:
            if name in shape_map:
                return cls._shape_product(shape_map[name][0])
            return 0

        def tensor_dtype_size(name: str) -> int:
            if name in shape_map:
                return cls._dtype_nbytes(shape_map[name][1])
            if name in initializer_map:
                return cls._dtype_nbytes(initializer_map[name].data_type)
            return 4

        output_elements = tensor_elements(node.output[0]) if node.output else 0
        output_bytes = output_elements * (tensor_dtype_size(node.output[0]) if node.output else 4)

        if op_type == "Conv" and len(node.input) >= 2:
            input_name = node.input[0]
            weight_name = node.input[1]
            input_shape = shape_map.get(input_name, ([1, 1, 1, 1], 1))[0]
            weight_shape = shape_map.get(weight_name, ([], 1))[0]
            cin = input_shape[1] if len(input_shape) > 1 else 1
            kernel_mul = 1
            if len(weight_shape) >= 3:
                for dim in weight_shape[2:]:
                    kernel_mul *= max(dim, 1)
            flops = 2.0 * output_elements * cin * kernel_mul
            bytes_io = (
                tensor_elements(input_name) * tensor_dtype_size(input_name)
                + tensor_elements(weight_name) * tensor_dtype_size(weight_name)
                + output_bytes
            )
            return flops, float(bytes_io)

        if op_type in {"Gemm", "MatMul"} and len(node.input) >= 2:
            a_name = node.input[0]
            b_name = node.input[1]
            a_shape = shape_map.get(a_name, ([1, 1], 1))[0]
            b_shape = shape_map.get(b_name, ([1, 1], 1))[0]
            if len(a_shape) >= 2 and len(b_shape) >= 2:
                m = a_shape[-2]
                k = a_shape[-1]
                n = b_shape[-1]
                flops = 2.0 * m * n * k
            else:
                flops = 2.0 * output_elements
            bytes_io = (
                tensor_elements(a_name) * tensor_dtype_size(a_name)
                + tensor_elements(b_name) * tensor_dtype_size(b_name)
                + output_bytes
            )
            return flops, float(bytes_io)

        if op_type in {"Relu", "Add", "Mul", "Sub", "Div", "Sigmoid", "Tanh", "BatchNormalization"}:
            flops = float(max(output_elements, 1))
            input_bytes = 0
            for in_name in node.input:
                input_bytes += tensor_elements(in_name) * tensor_dtype_size(in_name)
            return flops, float(input_bytes + output_bytes)

        # Fallback estimate for unknown operators.
        flops = float(max(output_elements, 1))
        input_bytes = 0
        for in_name in node.input:
            input_bytes += tensor_elements(in_name) * tensor_dtype_size(in_name)
        return flops, float(input_bytes + output_bytes)

    @classmethod
    def estimate_arithmetic_intensity(cls, model_path: Path) -> Tuple[float, float, float]:
        model = onnx.load(str(model_path))
        inferred = shape_inference.infer_shapes(model)

        shape_map = cls._extract_shapes(inferred)
        initializer_map = {init.name: init for init in inferred.graph.initializer}

        total_flops = 0.0
        total_bytes = 0.0
        for node in inferred.graph.node:
            flops, bytes_io = cls._estimate_node_flops_bytes(node, shape_map, initializer_map)
            total_flops += flops
            total_bytes += bytes_io

        ai = total_flops / max(total_bytes, 1.0)
        return ai, total_flops, total_bytes

    @staticmethod
    def count_nodes(model_path: Path) -> int:
        return len(onnx.load(str(model_path)).graph.node)

    @staticmethod
    def count_parameters(model_path: Path) -> int:
        model = onnx.load(str(model_path))
        total = 0
        for init in model.graph.initializer:
            size = 1
            for dim in init.dims:
                size *= int(dim)
            total += size
        return total


class RooflineAnalyzer:
    @staticmethod
    def classify(ai: float, peak_compute_gflops: float, peak_bandwidth_gbps: float) -> str:
        ridge_point = peak_compute_gflops / max(peak_bandwidth_gbps, 1e-9)
        return "memory-bound" if ai < ridge_point else "compute-bound"

    @staticmethod
    def attained_gflops(total_flops: float, latency_ms: float) -> float:
        return (total_flops / max(latency_ms / 1000.0, 1e-9)) / 1e9


class OptimizedModelExporter:
    @staticmethod
    def export(model_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.optimized_model_filepath = str(output_path)

        # Use CPU EP for export to avoid compiled nodes that cannot be serialized.
        ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return output_path


class ScalabilityVisualizer:
    @staticmethod
    def plot_speedup(df: pd.DataFrame, path: Path) -> None:
        if df.empty:
            return
        ordered = df.sort_values("params_million")
        plt.figure(figsize=(8, 5))
        plt.plot(ordered["params_million"], ordered["speedup_x"], marker="o", color="#1a936f")
        plt.xlabel("Model Size (Million Parameters)")
        plt.ylabel("Speedup (baseline/optimized)")
        plt.title("Fusion Benefit vs Model Size")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    @staticmethod
    def plot_latency(df: pd.DataFrame, path: Path) -> None:
        if df.empty:
            return
        ordered = df.sort_values("params_million")
        x = np.arange(len(ordered))
        width = 0.35

        plt.figure(figsize=(9, 5))
        plt.bar(x - width / 2, ordered["baseline_mean_ms"], width=width, label="Baseline", color="#c03221")
        plt.bar(x + width / 2, ordered["optimized_mean_ms"], width=width, label="Optimized", color="#0a9396")
        plt.xticks(x, ordered["model"], rotation=25, ha="right")
        plt.ylabel("Latency (ms)")
        plt.title("Latency Across Model Scale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()


class MultiModelScalabilityPipeline:
    def __init__(self, config: ScalabilityConfig) -> None:
        self.config = config
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

    def _run_single_repeat(self, model_path: Path, run_idx: int) -> pd.DataFrame:
        run_dir = self.config.results_dir / model_path.stem / f"run_{run_idx:02d}"
        runner = NPUBenchmarkRunner(
            BenchmarkConfig(
                model_path=model_path,
                results_dir=run_dir,
                iterations=self.config.iterations,
                warmup_iterations=self.config.warmup_iterations,
                random_seed=42 + run_idx,
            )
        )
        return runner.run()

    def _summarize_model(self, model_path: Path) -> Dict[str, Any]:
        baseline_samples: List[float] = []
        optimized_samples: List[float] = []

        for run_idx in range(1, self.config.repeats + 1):
            run_df = self._run_single_repeat(model_path, run_idx)
            baseline_samples.append(float(run_df.loc[run_df["mode"] == "baseline", "avg_latency_ms"].iloc[0]))
            optimized_samples.append(float(run_df.loc[run_df["mode"] == "optimized", "avg_latency_ms"].iloc[0]))

        baseline_mean = float(np.mean(baseline_samples))
        optimized_mean = float(np.mean(optimized_samples))
        baseline_std = float(np.std(baseline_samples, ddof=1)) if len(baseline_samples) > 1 else 0.0
        optimized_std = float(np.std(optimized_samples, ddof=1)) if len(optimized_samples) > 1 else 0.0

        z_score = 1.96
        baseline_ci95 = z_score * baseline_std / max(np.sqrt(len(baseline_samples)), 1)
        optimized_ci95 = z_score * optimized_std / max(np.sqrt(len(optimized_samples)), 1)

        optimized_model_path = self.config.results_dir / model_path.stem / "optimized_graph.onnx"
        OptimizedModelExporter.export(model_path, optimized_model_path)

        original_nodes = ONNXGraphMetrics.count_nodes(model_path)
        optimized_nodes = ONNXGraphMetrics.count_nodes(optimized_model_path)
        params = ONNXGraphMetrics.count_parameters(model_path)

        ai_baseline, flops_baseline, _ = ONNXGraphMetrics.estimate_arithmetic_intensity(model_path)
        ai_optimized, flops_optimized, _ = ONNXGraphMetrics.estimate_arithmetic_intensity(optimized_model_path)

        ridge_point = self.config.peak_compute_gflops / max(self.config.peak_bandwidth_gbps, 1e-9)

        return {
            "model": model_path.stem,
            "model_path": str(model_path),
            "repeats": self.config.repeats,
            "params": params,
            "params_million": params / 1_000_000.0,
            "original_nodes": original_nodes,
            "optimized_nodes": optimized_nodes,
            "node_reduction_pct": ((original_nodes - optimized_nodes) / max(original_nodes, 1)) * 100.0,
            "baseline_mean_ms": baseline_mean,
            "optimized_mean_ms": optimized_mean,
            "baseline_std_ms": baseline_std,
            "optimized_std_ms": optimized_std,
            "baseline_ci95_ms": baseline_ci95,
            "optimized_ci95_ms": optimized_ci95,
            "speedup_x": baseline_mean / max(optimized_mean, 1e-9),
            "latency_reduction_pct": ((baseline_mean - optimized_mean) / max(baseline_mean, 1e-9)) * 100.0,
            "c_factor_ratio": optimized_mean / max(baseline_mean, 1e-9),
            "baseline_ai_flop_per_byte": ai_baseline,
            "optimized_ai_flop_per_byte": ai_optimized,
            "ridge_point_flop_per_byte": ridge_point,
            "baseline_bound": RooflineAnalyzer.classify(
                ai_baseline,
                self.config.peak_compute_gflops,
                self.config.peak_bandwidth_gbps,
            ),
            "optimized_bound": RooflineAnalyzer.classify(
                ai_optimized,
                self.config.peak_compute_gflops,
                self.config.peak_bandwidth_gbps,
            ),
            "baseline_attained_gflops": RooflineAnalyzer.attained_gflops(flops_baseline, baseline_mean),
            "optimized_attained_gflops": RooflineAnalyzer.attained_gflops(flops_optimized, optimized_mean),
            "peak_compute_gflops": self.config.peak_compute_gflops,
            "peak_bandwidth_gbps": self.config.peak_bandwidth_gbps,
        }

    def run(self) -> pd.DataFrame:
        rows = [self._summarize_model(model) for model in self.config.models]
        df = pd.DataFrame(rows).sort_values("params")

        matrix_path = self.config.results_dir / "scalability_matrix.csv"
        df.to_csv(matrix_path, index=False)

        ScalabilityVisualizer.plot_speedup(df, self.config.results_dir / "scalability_speedup.png")
        ScalabilityVisualizer.plot_latency(df, self.config.results_dir / "scalability_latency.png")

        return df


def parse_args() -> ScalabilityConfig:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run multi-model benchmark repeats and produce scalability + roofline summary artifacts."
    )
    parser.add_argument(
        "--models-dir",
        default=str(project_root / "models"),
        help="Directory containing ONNX model files.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional explicit model file paths. If omitted, all *.onnx in models-dir are used.",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations per run.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per run.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated runs per model.")
    parser.add_argument(
        "--results-dir",
        default=str(project_root / "results"),
        help="Results root directory.",
    )
    parser.add_argument(
        "--peak-compute-gflops",
        type=float,
        default=1000.0,
        help="Hardware peak compute in GFLOP/s for roofline analysis.",
    )
    parser.add_argument(
        "--peak-bandwidth-gbps",
        type=float,
        default=30.0,
        help="Hardware memory bandwidth in GB/s for roofline analysis.",
    )

    args = parser.parse_args()

    models: List[Path]
    if args.models:
        models = [Path(path).resolve() for path in args.models]
    else:
        models_dir = Path(args.models_dir).resolve()
        models = sorted(models_dir.glob("*.onnx"))

    if not models:
        raise ValueError("No ONNX model files found. Provide --models or place .onnx files in models directory.")

    return ScalabilityConfig(
        models=models,
        results_dir=Path(args.results_dir).resolve(),
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        repeats=args.repeats,
        peak_compute_gflops=args.peak_compute_gflops,
        peak_bandwidth_gbps=args.peak_bandwidth_gbps,
    )


def main() -> None:
    config = parse_args()
    pipeline = MultiModelScalabilityPipeline(config)
    summary_df = pipeline.run()

    print("Scalability study complete.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
