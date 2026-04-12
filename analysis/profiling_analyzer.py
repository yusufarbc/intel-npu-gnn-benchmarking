from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class AnalyzeConfig:
    baseline_json: Path
    optimized_json: Path
    results_dir: Path
    top_k: int = 15
    top_speedup_k: int = 5


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
            category = event.get("cat")
            if isinstance(category, str) and category and category.lower() not in {"node", "op", "kernel", "session"}:
                if "op_name" not in (event.get("args") or {}):
                    continue
            rows.append({"mode": mode, "operator": op_name, "duration_us": duration_us})
        return rows


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
        aggregated.to_csv(self.config.results_dir / "operator_breakdown_by_mode.csv", index=False)
        comparison.to_csv(self.config.results_dir / "operator_breakdown.csv", index=False)
        
        self._save_chart(comparison)
        self._generate_summary(comparison)

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

    def _generate_summary(self, df: pd.DataFrame) -> None:
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

        print(f"Analysis saved to {self.config.results_dir}")


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
