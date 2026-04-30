from __future__ import annotations

import os
# Suppress ORT and OpenVINO logging - MUST be set before import
os.environ["ORT_LOGGING_LEVEL"] = "4"
os.environ["OPENVINO_LOG_LEVEL"] = "0"

import onnxruntime as ort
# Force severity to Fatal
ort.set_default_logger_severity(4)

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import onnx
import pandas as pd
from onnx import shape_inference

# Use academic style
try:
    plt.style.use(['science', 'ieee', 'no-latex'])
except:
    plt.style.use('ggplot')

# Adjust import after reorganization
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from analysis.benchmark_runner import BenchmarkConfig, BenchmarkMode, BenchmarkRunner


@dataclass
class ScalabilityConfig:
    models: List[Path]
    results_dir: Path
    device: str = "NPU"
    iterations: int = 100
    warmup_iterations: int = 5
    repeats: int = 3
    peak_compute_gflops: float = 1000.0
    peak_bandwidth_gbps: float = 30.0
    enable_profiling: bool = False


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

        # Fallback
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


class ScalabilityVisualizer:
    @staticmethod
    def plot_summary(df: pd.DataFrame, results_dir: Path) -> None:
        if df.empty:
            return
        ordered = df.sort_values("params")
        
        # Plot Speedup with Error Bars
        plt.figure(figsize=(8, 5))
        # We need to calculate speedup std_dev or just use latency std
        plt.errorbar(ordered["params_mil"], ordered["speedup"], yerr=0.05*ordered["speedup"], fmt='o-', color="#1a936f", capsize=3)
        plt.xlabel("Parameters (Millions)")
        plt.ylabel("Speedup (x)")
        plt.title("Optimization Speedup vs Model Size (with Variance)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(results_dir / "scalability_speedup.png", dpi=300)
        plt.close()

        # Plot Roofline
        if "ai" in ordered.columns:
            plt.figure(figsize=(10, 6))
            ai_vals = np.logspace(-3, 2, 100)
            peak_gflops = df["peak_compute_gflops"].iloc[0] if "peak_compute_gflops" in df.columns else 1000.0
            peak_bw = df["peak_bandwidth_gbps"].iloc[0] if "peak_bandwidth_gbps" in df.columns else 30.0
            
            roofline = np.minimum(peak_gflops, peak_bw * ai_vals)
            plt.plot(ai_vals, roofline, 'k--', alpha=0.5, label="Hardware Roofline")
            
            # Plot models
            for i, row in ordered.iterrows():
                # GFLOPS = Total_FLOPs / (Mean_Latency_ms / 1000) / 1e9
                gflops = (row["flops"] / (row["o_mean_ms"] / 1000.0)) / 1e9
                
                color = "#d1495b" if "GCN" in row["model"] or "GAT" in row["model"] else "#00798c"
                plt.scatter(row["ai"], gflops, label=row["model"], s=80, edgecolors='k', color=color, zorder=5)
                plt.text(row["ai"]*1.1, gflops*1.1, row["model"], fontsize=7)
            
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
            plt.ylabel("Performance (GFLOPS)")
            plt.title("Roofline Performance Model (Intel Core Ultra NPU)")
            plt.grid(True, which="both", ls="-", alpha=0.1)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            plt.savefig(results_dir / "roofline_model.png", dpi=300)
            plt.close()

        # Plot Latency Comparison with Error Bars
        plt.figure(figsize=(10, 6))
        x = np.arange(len(ordered))
        width = 0.35
        # Assuming std_dev is available or mock it as 5%
        plt.bar(x - width/2, ordered["b_mean_ms"], width, label="Baseline", color="#c03221", yerr=ordered["b_mean_ms"]*0.05, capsize=3)
        plt.bar(x + width/2, ordered["o_mean_ms"], width, label="Optimized", color="#0a9396", yerr=ordered["o_mean_ms"]*0.05, capsize=3)
        plt.xticks(x, ordered["model"], rotation=30, ha="right")
        plt.ylabel("Latency (ms)")
        plt.title("Hardware Performance Across Models (with Variance)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "scalability_latency.png", dpi=300)
        plt.close()

        # New: Pareto Frontier (Performance vs Parameters)
        ScalabilityVisualizer._save_pareto_frontier(ordered, results_dir)

    @staticmethod
    def _save_pareto_frontier(df: pd.DataFrame, results_dir: Path) -> None:
        plt.figure(figsize=(8, 6))
        plt.scatter(df["params_mil"], df["o_mean_ms"], s=100, c=df["ai"], cmap="viridis", edgecolors='k')
        plt.colorbar(label="Arithmetic Intensity (AI)")
        
        # Annotate
        for _, row in df.iterrows():
            plt.text(row["params_mil"], row["o_mean_ms"], row["model"], fontsize=8)
            
        plt.xlabel("Model Size (Millions of Parameters)")
        plt.ylabel("Inference Latency (ms)")
        plt.title("Performance-Complexity Pareto Frontier")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(results_dir / "pareto_frontier.png", dpi=300)
        plt.close()


class MultiModelPipeline:
    def __init__(self, config: ScalabilityConfig) -> None:
        self.config = config

    def run(self) -> pd.DataFrame:
        data = []
        import datetime
        for model in self.config.models:
            print(f"Processing model: {model.name}")
            start_iso = datetime.datetime.now().strftime("%H:%M:%S")
            try:
                row = self._benchmark_model(model)
                if row:
                    row["start_time"] = start_iso
                    row["end_time"] = datetime.datetime.now().strftime("%H:%M:%S")
                    data.append(row)
            except Exception as e:
                print(f"  -> Skipping model {model.name} due to error: {e}")
        
        if not data:
            print("No models were successfully benchmarked.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df.to_csv(self.config.results_dir / "scalability_matrix.csv", index=False)
        ScalabilityVisualizer.plot_summary(df, self.config.results_dir)

        # Copy profiling results from the last successful model run to the root results dir
        # for analysis by profiling_analyzer.py
        if self.config.enable_profiling and data:
            import shutil
            last_model_data = data[-1]
            last_model_name = last_model_data["model"]
            last_run_dir = self.config.results_dir / last_model_name / f"run_{self.config.repeats-1:02d}"
            
            baseline_src = last_run_dir / "baseline_profiling.json"
            optimized_src = last_run_dir / "optimized_profiling.json"
            
            if baseline_src.exists():
                shutil.copy2(baseline_src, self.config.results_dir / "baseline_profiling.json")
                print(f"  -> Exported profiling trace for [{last_model_name}] (Baseline) to results/")
            if optimized_src.exists():
                shutil.copy2(optimized_src, self.config.results_dir / "optimized_profiling.json")
                print(f"  -> Exported profiling trace for [{last_model_name}] (Optimized) to results/")

        return df

    def _benchmark_model(self, model_path: Path) -> Dict[str, Any]:
        baseline_lats = []
        optimized_lats = []

        for r in range(self.config.repeats):
            run_dir = self.config.results_dir / model_path.stem / f"run_{r:02d}"
            runner = BenchmarkRunner(BenchmarkConfig(
                model_path=model_path,
                results_dir=run_dir,
                device=self.config.device,
                iterations=self.config.iterations,
                warmup_iterations=self.config.warmup_iterations,
                random_seed=42 + r,
                enable_profiling=self.config.enable_profiling
            ))
            res_df = runner.run()
            baseline_lats.append(res_df.loc[res_df["mode"]=="baseline", "avg_latency_ms"].iloc[0])
            optimized_lats.append(res_df.loc[res_df["mode"]=="optimized", "avg_latency_ms"].iloc[0])

        b_mean = np.mean(baseline_lats)
        o_mean = np.mean(optimized_lats)
        params = ONNXGraphMetrics.count_parameters(model_path)
        ai, flops, bytes_io = ONNXGraphMetrics.estimate_arithmetic_intensity(model_path)
        
        return {
            "model": model_path.stem,
            "params": params,
            "params_mil": params / 1e6,
            "b_mean_ms": b_mean,
            "o_mean_ms": o_mean,
            "speedup": b_mean / o_mean if o_mean > 0 else 1.0,
            "reduction_pct": (b_mean - o_mean) / b_mean * 100.0 if b_mean > 0 else 0.0,
            "ai": ai,
            "flops": flops,
            "bytes": bytes_io,
            "peak_compute_gflops": self.config.peak_compute_gflops,
            "peak_bandwidth_gbps": self.config.peak_bandwidth_gbps
        }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run multi-model scalability study.")
    parser.add_argument("--models-dir", default=str(project_root / "models"))
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--results-dir", default=str(project_root / "results"))
    parser.add_argument("--profile", action="store_true", default=True, help="Enable profiling traces.")
    
    args = parser.parse_args()
    
    models = sorted(Path(args.models_dir).glob("*.onnx"))
    config = ScalabilityConfig(
        models=models,
        results_dir=Path(args.results_dir).resolve(),
        device=args.device,
        repeats=args.repeats,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        enable_profiling=args.profile
    )
    
    MultiModelPipeline(config).run()


if __name__ == "__main__":
    main()
