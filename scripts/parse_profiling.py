from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from parse_operator_breakdown import BreakdownConfig, OperatorBreakdownPipeline


@dataclass
class ProfilingReportConfig:
    baseline_json: Path
    optimized_json: Path
    results_dir: Path
    top_k: int = 15
    top_speedup_k: int = 5


class ProfilingReportBuilder:
    def __init__(self, config: ProfilingReportConfig) -> None:
        self.config = config
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        pipeline = OperatorBreakdownPipeline(
            BreakdownConfig(
                baseline_json=self.config.baseline_json,
                optimized_json=self.config.optimized_json,
                results_dir=self.config.results_dir,
                top_k=self.config.top_k,
            )
        )

        comparison_df = pipeline.run()

        if comparison_df.empty:
            print("No operator events were parsed from profiling JSON files.")
            return

        # In Big-O terms, fusion usually keeps arithmetic complexity class unchanged,
        # but changes hidden constants by reducing memory traffic and launch overhead.
        top_speedup_df = (
            comparison_df[comparison_df["baseline_total_ms"] > 0]
            .sort_values("reduction_pct", ascending=False)
            .head(self.config.top_speedup_k)
            .copy()
        )

        disappeared_df = comparison_df[
            (comparison_df["baseline_count"] > 0) & (comparison_df["optimized_count"] == 0)
        ].copy()

        count_delta_df = pd.DataFrame(
            [
                {
                    "metric": "operator_invocation_count",
                    "baseline": float(comparison_df["baseline_count"].sum()),
                    "optimized": float(comparison_df["optimized_count"].sum()),
                }
            ]
        )
        count_delta_df["delta"] = count_delta_df["optimized"] - count_delta_df["baseline"]
        count_delta_df["reduction_pct"] = count_delta_df.apply(
            lambda row: ((row["baseline"] - row["optimized"]) / row["baseline"]) * 100.0
            if row["baseline"] > 0
            else 0.0,
            axis=1,
        )

        top_speedup_path = self.config.results_dir / "operator_top5_speedup.csv"
        disappeared_path = self.config.results_dir / "disappeared_operators.csv"
        count_delta_path = self.config.results_dir / "operator_count_delta.csv"

        top_speedup_df.to_csv(top_speedup_path, index=False)
        disappeared_df.to_csv(disappeared_path, index=False)
        count_delta_df.to_csv(count_delta_path, index=False)

        print("Profiling parse complete.")
        print(f"Saved: {top_speedup_path}")
        print(f"Saved: {disappeared_path}")
        print(f"Saved: {count_delta_path}")
        print("Top accelerated operators:")
        print(top_speedup_df.to_string(index=False))


def parse_args() -> ProfilingReportConfig:
    project_root = Path(__file__).resolve().parent.parent
    default_results_dir = project_root / "results"

    parser = argparse.ArgumentParser(
        description="Parse ONNX Runtime profiling files and report operator-level acceleration evidence."
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
        help="Output directory for CSV and PNG files.",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Top-k operators in comparison chart.")
    parser.add_argument("--top-speedup-k", type=int, default=5, help="Top-k accelerated operators.")

    args = parser.parse_args()

    return ProfilingReportConfig(
        baseline_json=Path(args.baseline_json).resolve(),
        optimized_json=Path(args.optimized_json).resolve(),
        results_dir=Path(args.results_dir).resolve(),
        top_k=args.top_k,
        top_speedup_k=args.top_speedup_k,
    )


def main() -> None:
    config = parse_args()
    ProfilingReportBuilder(config).run()


if __name__ == "__main__":
    main()
