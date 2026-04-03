from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
    project_root: Path
    models: List[Path]
    results_dir: Path
    figures_dir: Path
    iterations: int = 100
    warmup: int = 5
    repeats: int = 3
    peak_compute_gflops: float = 1000.0
    peak_bandwidth_gbps: float = 30.0
    top_k: int = 15
    top_speedup_k: int = 5


class CommandRunner:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def run(self, command: List[str]) -> None:
        printable = " ".join(command)
        print(f"[run_all] Running: {printable}")
        subprocess.run(command, cwd=self.cwd, check=True)


class FigureCollector:
    @staticmethod
    def copy_results_pngs(results_dir: Path, figures_dir: Path) -> List[Path]:
        figures_dir.mkdir(parents=True, exist_ok=True)
        copied: List[Path] = []

        for png_file in results_dir.rglob("*.png"):
            destination = figures_dir / png_file.name
            shutil.copy2(png_file, destination)
            copied.append(destination)

        return copied


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.runner = CommandRunner(cwd=self.config.project_root)

    def _run_single_model_benchmark(self, model_path: Path) -> None:
        self.runner.run(
            [
                sys.executable,
                "scripts/benchmark_npu.py",
                "--model",
                str(model_path),
                "--iterations",
                str(self.config.iterations),
                "--results-dir",
                str(self.config.results_dir),
            ]
        )

    def _run_profiling_parser(self) -> None:
        self.runner.run(
            [
                sys.executable,
                "scripts/parse_profiling.py",
                "--results-dir",
                str(self.config.results_dir),
                "--top-k",
                str(self.config.top_k),
                "--top-speedup-k",
                str(self.config.top_speedup_k),
            ]
        )

    def _run_scalability(self) -> None:
        command = [
            sys.executable,
            "scripts/run_scalability_study.py",
            "--results-dir",
            str(self.config.results_dir),
            "--iterations",
            str(self.config.iterations),
            "--warmup",
            str(self.config.warmup),
            "--repeats",
            str(self.config.repeats),
            "--peak-compute-gflops",
            str(self.config.peak_compute_gflops),
            "--peak-bandwidth-gbps",
            str(self.config.peak_bandwidth_gbps),
            "--models",
        ]
        command.extend(str(model) for model in self.config.models)
        self.runner.run(command)

    def _copy_figures(self) -> None:
        copied = FigureCollector.copy_results_pngs(self.config.results_dir, self.config.figures_dir)
        print(f"[run_all] Copied {len(copied)} figure(s) to: {self.config.figures_dir}")

    def run(self) -> None:
        # Pipeline intent: produce reproducible evidence from raw benchmark to paper-ready figures.
        first_model = self.config.models[0]
        print(f"[run_all] Using first model for single benchmark: {first_model}")
        self._run_single_model_benchmark(first_model)
        self._run_profiling_parser()
        self._run_scalability()
        self._copy_figures()
        print("[run_all] Pipeline complete.")


def parse_args() -> PipelineConfig:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Run full benchmark pipeline: benchmark, parse profiling, scalability study, and figure copy."
    )
    parser.add_argument(
        "--models-dir",
        default=str(project_root / "models"),
        help="Directory containing ONNX models.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional explicit list of ONNX model files. If omitted, all *.onnx in models-dir are used.",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Inference iterations for benchmark/scalability.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for scalability script.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeat count per model for scalability.")
    parser.add_argument(
        "--results-dir",
        default=str(project_root / "results"),
        help="Directory for benchmark and analysis outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(project_root / "paper" / "figures"),
        help="Directory to copy PNG artifacts for paper.",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Top-k operator chart limit.")
    parser.add_argument("--top-speedup-k", type=int, default=5, help="Top-k accelerated operators in report.")
    parser.add_argument(
        "--peak-compute-gflops",
        type=float,
        default=1000.0,
        help="Roofline peak compute GFLOP/s.",
    )
    parser.add_argument(
        "--peak-bandwidth-gbps",
        type=float,
        default=30.0,
        help="Roofline peak memory bandwidth GB/s.",
    )

    args = parser.parse_args()

    if args.models:
        models = [Path(model).resolve() for model in args.models]
    else:
        models = sorted(Path(args.models_dir).resolve().glob("*.onnx"))

    if not models:
        raise ValueError("No ONNX models found. Add models under models/ or pass --models explicitly.")

    return PipelineConfig(
        project_root=project_root,
        models=models,
        results_dir=Path(args.results_dir).resolve(),
        figures_dir=Path(args.figures_dir).resolve(),
        iterations=args.iterations,
        warmup=args.warmup,
        repeats=args.repeats,
        peak_compute_gflops=args.peak_compute_gflops,
        peak_bandwidth_gbps=args.peak_bandwidth_gbps,
        top_k=args.top_k,
        top_speedup_k=args.top_speedup_k,
    )


def main() -> None:
    config = parse_args()
    PipelineRunner(config).run()


if __name__ == "__main__":
    main()
