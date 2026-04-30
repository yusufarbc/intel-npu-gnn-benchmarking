from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import pandas as pd

# Use academic style
try:
    plt.style.use(['science', 'ieee', 'no-latex'])
except:
    plt.style.use('ggplot') # Fallback


@dataclass
class AnalyzeConfig:
    baseline_json: Path
    optimized_json: Path
    results_dir: Path
    top_k: int = 15
    top_speedup_k: int = 5
    compilation_event_name: str = "session_initialization"
    total_run_event_name: str = "model_run"


class ProfilingTraceLoader:
    @staticmethod
    def load_events(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Profiling file not found: {path}")

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if isinstance(payload, list):
            return [event for event in payload if isinstance(event, dict)]

        if isinstance(payload, dict):
            trace_events = payload.get("traceEvents", [])
            if isinstance(trace_events, list):
                return [event for event in trace_events if isinstance(event, dict)]

        return []


class OperatorEventParser:
    @staticmethod
    def _extract_operator_name(event: Dict[str, Any]) -> Optional[str]:
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        op_name = args.get("op_name") or event.get("op_name")
        if isinstance(op_name, str) and op_name.strip():
            return op_name.strip()

        raw_name = event.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None

        cleaned = raw_name.strip()
        if cleaned.endswith("_kernel_time"):
            cleaned = cleaned.replace("_kernel_time", "")
        if "(" in cleaned:
            cleaned = cleaned.split("(", 1)[0].strip()
        if "::" in cleaned:
            cleaned = cleaned.split("::")[-1].strip()
        return cleaned or None

    @staticmethod
    def _extract_duration_us(event: Dict[str, Any]) -> Optional[float]:
        duration = event.get("dur")
        if isinstance(duration, (int, float)) and duration >= 0:
            return float(duration)
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        for key in ["dur", "duration", "duration_us", "op_time_us"]:
            value = args.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                return float(value)
        return None

    @classmethod
    def parse_operator_rows(cls, events: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for event in events:
            op_name = cls._extract_operator_name(event)
            duration_us = cls._extract_duration_us(event)
            if op_name is None or duration_us is None:
                continue
            
            # Extract provider from args
            args = event.get("args", {})
            provider = args.get("provider", "Unknown")
            
            category = "compute"
            name_lower = op_name.lower()
            if any(x in name_lower for x in ["memcpy", "dma", "copy", "transfer"]):
                category = "dma"
            
            rows.append({
                "mode": mode, 
                "operator": op_name, 
                "duration_us": duration_us,
                "category": category,
                "provider": provider
            })
        return rows

    @staticmethod
    def extract_compilation_time_ms(events: List[Dict[str, Any]], event_name: str) -> float:
        for event in events:
            if event.get("name") == event_name:
                return float(event.get("dur", 0)) / 1000.0
        return 0.0

    @staticmethod
    def extract_total_inference_time_ms(events: List[Dict[str, Any]], event_name: str) -> float:
        # model_run events occur multiple times, we average them
        runs = [float(e.get("dur", 0)) / 1000.0 for e in events if e.get("name") == event_name]
        return float(np.mean(runs)) if runs else 0.0


class ProfilingAnalyzer:
    def __init__(self, config: AnalyzeConfig) -> None:
        self.config = config
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        baseline_events = ProfilingTraceLoader.load_events(self.config.baseline_json)
        optimized_events = ProfilingTraceLoader.load_events(self.config.optimized_json)

        rows: List[Dict[str, Any]] = []
        rows.extend(OperatorEventParser.parse_operator_rows(baseline_events, mode="baseline"))
        rows.extend(OperatorEventParser.parse_operator_rows(optimized_events, mode="optimized"))

        if not rows:
            print("No operator events found.")
            return

        df = pd.DataFrame(rows)
        
        # Calculate Provider Distribution (CPU Fallback Analysis)
        provider_stats = df.groupby(["mode", "provider"])["duration_us"].agg(["sum", "count"]).unstack(fill_value=0)
        provider_stats.to_csv(self.config.results_dir / "provider_distribution.csv")
        
        # Calculate NPU vs CPU Support Ratio
        def get_support_ratio(mode_name: str) -> float:
            mode_df = df[df["mode"] == mode_name]
            if mode_df.empty: return 0.0
            npu_ops = mode_df[mode_df["provider"].str.contains("OpenVINO", na=False)]
            return (len(npu_ops) / len(mode_df)) * 100.0 if len(mode_df) > 0 else 0.0

        npu_ratio_base = get_support_ratio("baseline")
        npu_ratio_opt = get_support_ratio("optimized")
        
        # Calculate Category Breakdown
        breakdown = df.groupby(["mode", "category"])["duration_us"].sum().unstack(fill_value=0) / 1000.0
        
        # Total Inference Times from trace
        t_base_ms = OperatorEventParser.extract_total_inference_time_ms(baseline_events, self.config.total_run_event_name)
        t_opt_ms = OperatorEventParser.extract_total_inference_time_ms(optimized_events, self.config.total_run_event_name)
        
        # Compilation Times
        c_base_ms = OperatorEventParser.extract_compilation_time_ms(baseline_events, self.config.compilation_event_name)
        c_opt_ms = OperatorEventParser.extract_compilation_time_ms(optimized_events, self.config.compilation_event_name)

        # FGR: Fusion Gain Ratio
        fgr = t_base_ms / t_opt_ms if t_opt_ms > 0 else 0.0
        
        # CEI: Compilation Efficiency Index (Compilation time / latency reduction)
        reduction_ms = max(0, t_base_ms - t_opt_ms)
        cei = c_opt_ms / reduction_ms if reduction_ms > 0 else 0.0

        # Detailed breakdown including dispatch
        # Dispatch = Total - (Compute + DMA)
        stats = []
        for mode, t_total in [("baseline", t_base_ms), ("optimized", t_opt_ms)]:
            mode_df = df[df["mode"] == mode]
            compute_ms = mode_df[mode_df["category"] == "compute"]["duration_us"].sum() / 1000.0
            dma_ms = mode_df[mode_df["category"] == "dma"]["duration_us"].sum() / 1000.0
            # Note: compute_ms in traces is usually the sum of all kernels. 
            # If kernels run sequentially, this matches.
            dispatch_ms = max(0, t_total - (compute_ms + dma_ms))
            stats.append({
                "mode": mode,
                "total_ms": t_total,
                "compute_ms": compute_ms,
                "dma_ms": dma_ms,
                "dispatch_ms": dispatch_ms,
                "compilation_ms": c_base_ms if mode == "baseline" else c_opt_ms
            })
        
        stats_df = pd.DataFrame(stats)
        
        # Original Operator breakdown
        aggregated = (
            df.groupby(["mode", "operator"], as_index=False)
            .agg(count=("duration_us", "size"), total_us=("duration_us", "sum"), avg_us=("duration_us", "mean"))
            .sort_values(["mode", "total_us"], ascending=[True, False])
        )
        aggregated["total_ms"] = aggregated["total_us"] / 1000.0
        aggregated["avg_ms"] = aggregated["avg_us"] / 1000.0

        # Build comparison
        baseline = aggregated[aggregated["mode"] == "baseline"][
            ["operator", "count", "total_ms", "avg_ms"]
        ].rename(columns={"count": "b_count", "total_ms": "b_total_ms", "avg_ms": "b_avg_ms"})
        
        optimized = aggregated[aggregated["mode"] == "optimized"][
            ["operator", "count", "total_ms", "avg_ms"]
        ].rename(columns={"count": "o_count", "total_ms": "o_total_ms", "avg_ms": "o_avg_ms"})

        comparison = baseline.merge(optimized, on="operator", how="outer").fillna(0.0)
        comparison["reduction_pct"] = comparison.apply(
            lambda r: ((r["b_total_ms"] - r["o_total_ms"]) / r["b_total_ms"]) * 100.0 if r["b_total_ms"] > 0 else 0.0,
            axis=1
        )
        comparison = comparison.sort_values("b_total_ms", ascending=False)

        # Save artifacts
        stats_df.to_csv(self.config.results_dir / "research_metrics.csv", index=False)
        aggregated.to_csv(self.config.results_dir / "operator_breakdown_by_mode.csv", index=False)
        comparison.to_csv(self.config.results_dir / "operator_breakdown.csv", index=False)
        
        # Save advanced metrics to a separate JSON/Text
        metrics_summary = {
            "FGR": fgr,
            "CEI": cei,
            "FusionGainRatio": fgr,
            "CompilationEfficiencyIndex": cei,
            "LatencyReduction_ms": reduction_ms,
            "Baseline_Total_ms": t_base_ms,
            "Optimized_Total_ms": t_opt_ms,
            "NPU_Support_Ratio_Baseline_Pct": npu_ratio_base,
            "NPU_Support_Ratio_Optimized_Pct": npu_ratio_opt
        }
        with open(self.config.results_dir / "advanced_metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=4)

        self._save_chart(comparison)
        self._save_breakdown_chart(stats_df)
        self._save_latency_stacked_bar(stats_df)
        self._save_fgr_diverging_chart(metrics_summary)
        self._save_provider_chart(df)
        self._generate_summary(comparison, metrics_summary)

    def _save_latency_stacked_bar(self, df: pd.DataFrame) -> None:
        """100% Stacked Bar Chart for Latency Breakdown."""
        plt.figure(figsize=(8, 5))
        
        # Normalize to percentages
        pivot_df = df.set_index("mode")[["compute_ms", "dma_ms", "dispatch_ms"]]
        pivot_df_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        
        ax = pivot_df_pct.plot(kind="barh", stacked=True, color=["#264653", "#2a9d8f", "#e9c46a"])
        
        plt.xlabel("Percentage of Total Latency (%)")
        plt.title("Latency Breakdown: Where is time spent?")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.config.results_dir / "latency_stacked_100pct.png", dpi=300)
        plt.close()

    def _save_fgr_diverging_chart(self, metrics: Dict[str, Any]) -> None:
        """Diverging Bar Chart for Fusion Gain Ratio (FGR)."""
        fgr = metrics.get("FGR", 1.0)
        plt.figure(figsize=(6, 4))
        
        color = "forestgreen" if fgr >= 1.0 else "crimson"
        plt.barh(["GCN (Target)"], [fgr - 1.0], left=1.0, color=color)
        
        plt.axvline(1.0, color='black', linestyle='--', alpha=0.7)
        plt.xlabel("Fusion Gain Ratio (FGR)")
        plt.title("Optimization Efficiency (Baseline = 1.0)")
        plt.xlim(max(0, fgr - 1.5), fgr + 0.5)
        plt.tight_layout()
        plt.savefig(self.config.results_dir / "fgr_diverging.png", dpi=300)
        plt.close()

    def _save_provider_chart(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 6))
        # Pivot for provider distribution by duration
        pivot_df = df.groupby(["mode", "provider"])["duration_us"].sum().unstack(fill_value=0) / 1000.0
        pivot_df.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="viridis")
        
        plt.ylabel("Total Latency (ms)")
        plt.title("Execution Provider Distribution: NPU vs CPU Fallback")
        plt.legend(title="Provider", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.config.results_dir / "provider_fallback_analysis.png", dpi=200)
        plt.close()

    def _save_breakdown_chart(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 6))
        modes = df["mode"].tolist()
        compute = df["compute_ms"].tolist()
        dma = df["dma_ms"].tolist()
        dispatch = df["dispatch_ms"].tolist()

        plt.bar(modes, compute, label="Compute (Kernels)", color="#264653")
        plt.bar(modes, dma, bottom=compute, label="DMA (Transfer)", color="#2a9d8f")
        plt.bar(modes, dispatch, bottom=[c + d for c, d in zip(compute, dma)], label="Dispatch Overhead", color="#e9c46a")

        plt.ylabel("Latency (ms)")
        plt.title("Latency Breakdown: Compute vs DMA vs Dispatch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.results_dir / "latency_breakdown.png", dpi=200)
        plt.close()

    def _generate_summary(self, df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        top_speedup = df[df["b_total_ms"] > 0].sort_values("reduction_pct", ascending=False).head(self.config.top_speedup_k)
        disappeared = df[(df["b_count"] > 0) & (df["o_count"] == 0)]
        
        count_delta = pd.DataFrame([{
            "metric": "invocations",
            "baseline": df["b_count"].sum(),
            "optimized": df["o_count"].sum(),
        }])
        count_delta["reduction_pct"] = (count_delta["baseline"] - count_delta["optimized"]) / count_delta["baseline"] * 100.0

        top_speedup.to_csv(self.config.results_dir / "operator_top5_speedup.csv", index=False)
        disappeared.to_csv(self.config.results_dir / "disappeared_operators.csv", index=False)
        count_delta.to_csv(self.config.results_dir / "operator_count_delta.csv", index=False)

        print("\n" + "="*40)
        print("RESEARCH METRICS SUMMARY")
        print("-" * 40)
        print(f"Fusion Gain Ratio (FGR): {metrics['FGR']:.4f}")
        print(f"Compilation Efficiency Index (CEI): {metrics['CEI']:.4f}")
        print(f"NPU Support Ratio (Optimized): {metrics['NPU_Support_Ratio_Optimized_Pct']:.1f}%")
        print(f"Baseline Latency: {metrics['Baseline_Total_ms']:.2f} ms")
        print(f"Optimized Latency: {metrics['Optimized_Total_ms']:.2f} ms")
        print(f"Latency Reduction: {metrics['LatencyReduction_ms']:.2f} ms")
        print("=" * 40 + "\n")
        print(f"Detailed analysis saved to {self.config.results_dir}")

    def _save_chart(self, df: pd.DataFrame) -> None:
        top_df = df.head(self.config.top_k)
        plt.figure(figsize=(10, 6))
        x = range(len(top_df))
        width = 0.35
        plt.bar([i - width/2 for i in x], top_df["b_total_ms"], width, label="Baseline", color="#d1495b")
        plt.bar([i + width/2 for i in x], top_df["o_total_ms"], width, label="Optimized", color="#00798c")
        plt.xticks(x, top_df["operator"], rotation=45, ha="right")
        plt.ylabel("Total Time (ms)")
        plt.title("Operator Breakdown: Baseline vs Optimized")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.results_dir / "operator_breakdown_topk.png", dpi=200)
        plt.close()



def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"

    parser = argparse.ArgumentParser(description="Analyze ONNX Runtime profiling traces.")
    parser.add_argument("--baseline", default=str(results_dir / "baseline_profiling.json"))
    parser.add_argument("--optimized", default=str(results_dir / "optimized_profiling.json"))
    parser.add_argument("--results-dir", default=str(results_dir))
    parser.add_argument("--top-k", type=int, default=15)
    
    args = parser.parse_args()
    
    config = AnalyzeConfig(
        baseline_json=Path(args.baseline).resolve(),
        optimized_json=Path(args.optimized).resolve(),
        results_dir=Path(args.results_dir).resolve(),
        top_k=args.top_k
    )
    
    ProfilingAnalyzer(config).run()


if __name__ == "__main__":
    main()
