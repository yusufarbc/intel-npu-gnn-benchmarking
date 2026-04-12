from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import sys

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from engine.benchmark_runner import BenchmarkConfig, BenchmarkRunner


class HWComparator:
    def __init__(self, model_path: Path, results_dir: Path, iterations: int = 100, repeats: int = 3):
        self.model_path = model_path
        self.results_dir = results_dir / "hw_comparison" / model_path.stem
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.iterations = iterations
        self.repeats = repeats

    def run(self):
        import datetime
        import numpy as np
        
        start_time_total = datetime.datetime.now()
        print(f"[{start_time_total.strftime('%H:%M:%S')}] Starting 3-way hardware comparison for: {self.model_path.name}")
        print(f"Configuration: {self.repeats} repeats of {self.iterations} iterations each.")
        
        devices = ["CPU", "GPU", "NPU"]
        results = []

        for device in devices:
            start_device = datetime.datetime.now()
            print(f"[{start_device.strftime('%H:%M:%S')}] Testing device: {device}...")
            
            device_latencies = []
            device_memories = []
            device_cpus = []
            
            for r in range(self.repeats):
                print(f"  -> Repeat {r+1}/{self.repeats}...")
                device_results_dir = self.results_dir / device / f"run_{r:02d}"
                config = BenchmarkConfig(
                    model_path=self.model_path,
                    results_dir=device_results_dir,
                    device=device,
                    iterations=self.iterations
                )
                runner = BenchmarkRunner(config)
                df = runner.run()
                
                # Extract metrics
                opt_row = df.loc[df["mode"] == "optimized"].iloc[0]
                opt_lat = opt_row["avg_latency_ms"]
                peak_mem = opt_row["peak_memory_mb"]
                cpu_util = opt_row["cpu_utilization_pct"]
                
                device_latencies.append(opt_lat)
                device_memories.append(peak_mem)
                device_cpus.append(cpu_util)

            # Average across repeats
            avg_lat = float(np.mean(device_latencies))
            std_lat = float(np.std(device_latencies))
            avg_mem = float(np.mean(device_memories))
            avg_cpu = float(np.mean(device_cpus))
            
            results.append({
                "device": device,
                "latency_ms": avg_lat,
                "std_ms": std_lat,
                "throughput_ips": 1000.0 / avg_lat if avg_lat > 0 else 0,
                "peak_memory_mb": avg_mem,
                "cpu_util_pct": avg_cpu
            })
            
            end_device = datetime.datetime.now()
            duration_device = (end_device - start_device).total_seconds()
            print(f"[{end_device.strftime('%H:%M:%S')}] Device {device} completed ({self.repeats} runs) in {duration_device:.2f} seconds.")

        summary_df = pd.DataFrame(results)
        summary_df.to_csv(self.results_dir / "hw_comparison_summary.csv", index=False)
        
        self._plot_results(summary_df)
        
        end_time_total = datetime.datetime.now()
        duration_total = (end_time_total - start_time_total).total_seconds()
        print(f"[{end_time_total.strftime('%H:%M:%S')}] Hardware comparison complete.")
        print(f"Total time: {duration_total:.2f} seconds.")
        print(f"Results saved in: {self.results_dir}")
        return summary_df

    def _plot_results(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        colors = ["#264653", "#2a9d8f", "#e9c46a"]
        
        # Plot Latency
        plt.subplot(1, 2, 1)
        bars = plt.bar(df["device"], df["latency_ms"], yerr=df.get("std_ms", 0), capsize=5, color=colors)
        plt.ylabel("Latency (ms)")
        plt.title("Latency (Lower is Better)")
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom')

        # Plot Throughput
        plt.subplot(1, 2, 2)
        bars = plt.bar(df["device"], df["throughput_ips"], color=colors)
        plt.ylabel("Inferences / Second")
        plt.title("Throughput (Higher is Better)")
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.1f}", ha='center', va='bottom')

        plt.suptitle(f"Hardware Comparison: {self.model_path.name}\n(Mean of {self.repeats} runs)")
        plt.tight_layout()
        chart_name = f"hw_compare_{self.model_path.stem}.png"
        plt.savefig(self.results_dir / chart_name, dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run 3-way CPU vs GPU vs NPU comparison.")
    parser.add_argument("--model", required=True, help="Path to ONNX model.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per device.")
    
    args = parser.parse_args()
    
    model_path = Path(args.model).resolve()
    results_root = Path(__file__).resolve().parent.parent / "results"
    
    comparator = HWComparator(model_path, results_root, args.iterations, args.repeats)
    comparator.run()


if __name__ == "__main__":
    main()
