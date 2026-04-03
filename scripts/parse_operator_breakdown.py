from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class BreakdownConfig:
    baseline_json: Path
    optimized_json: Path
    results_dir: Path
    top_k: int = 15


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
            if isinstance(category, str) and category and category.lower() not in {
                "node",
                "op",
                "kernel",
                "session",
            }:
                if "op_name" not in (event.get("args") or {}):
                    continue

            rows.append(
                {
                    "mode": mode,
                    "operator": op_name,
                    "duration_us": duration_us,
                }
            )

        return rows


class BreakdownAnalyzer:
    @staticmethod
    def aggregate(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["mode", "operator", "count", "total_duration_ms", "avg_duration_ms"])

        df = pd.DataFrame(rows)
        grouped = (
            df.groupby(["mode", "operator"], as_index=False)
            .agg(count=("duration_us", "size"), total_duration_us=("duration_us", "sum"), avg_duration_us=("duration_us", "mean"))
            .sort_values(["mode", "total_duration_us"], ascending=[True, False])
        )

        grouped["total_duration_ms"] = grouped["total_duration_us"] / 1000.0
        grouped["avg_duration_ms"] = grouped["avg_duration_us"] / 1000.0

        return grouped[["mode", "operator", "count", "total_duration_ms", "avg_duration_ms"]]

    @staticmethod
    def build_comparison(aggregated: pd.DataFrame) -> pd.DataFrame:
        baseline = aggregated[aggregated["mode"] == "baseline"][
            ["operator", "count", "total_duration_ms", "avg_duration_ms"]
        ].rename(
            columns={
                "count": "baseline_count",
                "total_duration_ms": "baseline_total_ms",
                "avg_duration_ms": "baseline_avg_ms",
            }
        )

        optimized = aggregated[aggregated["mode"] == "optimized"][
            ["operator", "count", "total_duration_ms", "avg_duration_ms"]
        ].rename(
            columns={
                "count": "optimized_count",
                "total_duration_ms": "optimized_total_ms",
                "avg_duration_ms": "optimized_avg_ms",
            }
        )

        comparison = baseline.merge(optimized, on="operator", how="outer").fillna(0.0)
        comparison["delta_total_ms"] = comparison["optimized_total_ms"] - comparison["baseline_total_ms"]

        # Node merging leaves arithmetic complexity close to O(N+E) but often lowers memory traffic constants.
        # The ratio below is an empirical proxy for the hidden constant c in T(N) = c * O(N+E).
        comparison["reduction_pct"] = comparison.apply(
            lambda row: (
                ((row["baseline_total_ms"] - row["optimized_total_ms"]) / row["baseline_total_ms"]) * 100.0
                if row["baseline_total_ms"] > 0
                else 0.0
            ),
            axis=1,
        )

        comparison = comparison.sort_values("baseline_total_ms", ascending=False)
        return comparison


class BreakdownVisualizer:
    @staticmethod
    def save_topk_chart(comparison_df: pd.DataFrame, output_path: Path, top_k: int) -> None:
        if comparison_df.empty:
            return

        top_df = comparison_df.head(top_k).copy()
        labels = top_df["operator"].astype(str).tolist()

        x_positions = range(len(labels))
        width = 0.38

        plt.figure(figsize=(max(10, len(labels) * 0.75), 5.5))
        plt.bar(
            [x - width / 2 for x in x_positions],
            top_df["baseline_total_ms"],
            width=width,
            label="Baseline",
            color="#d1495b",
        )
        plt.bar(
            [x + width / 2 for x in x_positions],
            top_df["optimized_total_ms"],
            width=width,
            label="Optimized",
            color="#00798c",
        )

        plt.xticks(list(x_positions), labels, rotation=45, ha="right")
        plt.ylabel("Total Operator Time (ms)")
        plt.xlabel("Operator")
        plt.title("Top Operators: Baseline vs Optimized")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=220)
        plt.close()


class OperatorBreakdownPipeline:
    def __init__(self, config: BreakdownConfig) -> None:
        self.config = config
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        baseline_events = ProfilingTraceLoader.load_events(self.config.baseline_json)
        optimized_events = ProfilingTraceLoader.load_events(self.config.optimized_json)

        rows: List[Dict[str, Any]] = []
        rows.extend(OperatorEventParser.parse_operator_rows(baseline_events, mode="baseline"))
        rows.extend(OperatorEventParser.parse_operator_rows(optimized_events, mode="optimized"))

        aggregated = BreakdownAnalyzer.aggregate(rows)
        comparison = BreakdownAnalyzer.build_comparison(aggregated)

        aggregated_path = self.config.results_dir / "operator_breakdown_by_mode.csv"
        comparison_path = self.config.results_dir / "operator_breakdown.csv"
        chart_path = self.config.results_dir / "operator_breakdown_topk.png"

        aggregated.to_csv(aggregated_path, index=False)
        comparison.to_csv(comparison_path, index=False)
        BreakdownVisualizer.save_topk_chart(comparison, chart_path, self.config.top_k)

        return comparison


def parse_args() -> BreakdownConfig:
    project_root = Path(__file__).resolve().parent.parent
    default_results_dir = project_root / "results"

    parser = argparse.ArgumentParser(
        description="Parse ONNX Runtime profiling JSON files into operator-level breakdown CSV outputs."
    )
    parser.add_argument(
        "--baseline-json",
        default=str(default_results_dir / "baseline_profiling.json"),
        help="Path to baseline profiling JSON file.",
    )
    parser.add_argument(
        "--optimized-json",
        default=str(default_results_dir / "optimized_profiling.json"),
        help="Path to optimized profiling JSON file.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(default_results_dir),
        help="Output directory for CSV/PNG artifacts.",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Number of operators in comparison chart.")

    args = parser.parse_args()

    return BreakdownConfig(
        baseline_json=Path(args.baseline_json).resolve(),
        optimized_json=Path(args.optimized_json).resolve(),
        results_dir=Path(args.results_dir).resolve(),
        top_k=args.top_k,
    )


def main() -> None:
    config = parse_args()
    pipeline = OperatorBreakdownPipeline(config)
    comparison_df = pipeline.run()

    print("Operator-level breakdown complete.")
    print(comparison_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
