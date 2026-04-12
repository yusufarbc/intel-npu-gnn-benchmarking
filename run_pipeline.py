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
    top_k: int = 15
    top_speedup_k: int = 5


class CommandRunner:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def run(self, command: List[str]) -> None:
        printable = " ".join(command)
        print(f"[pipeline] Running: {printable}")
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

    def run(self) -> None:
        import datetime
        start_time = datetime.datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] Starting full benchmark pipeline...")

        # 1. Run single model benchmark (for profiling detail)
        first_model = self.config.models[0]
        print(f"[pipeline] Using first model for profiling: {first_model}")
        self.runner.run([
            sys.executable, "engine/benchmark_runner.py",
            "--model", str(first_model),
            "--iterations", str(self.config.iterations),
            "--results-dir", str(self.config.results_dir),
            "--profile"
        ])

        # 2. Run profiling analyzer
        self.runner.run([
            sys.executable, "analysis/profiling_analyzer.py",
            "--results-dir", str(self.config.results_dir),
            "--top-k", str(self.config.top_k)
        ])

        # 3. Run scalability study
        command = [
            sys.executable, "analysis/scalability_analyzer.py",
            "--results-dir", str(self.config.results_dir),
            "--iterations", str(self.config.iterations),
            "--warmup", str(self.config.warmup),
            "--repeats", str(self.config.repeats),
            "--models-dir", str(self.config.project_root / "models")
        ]
        self.runner.run(command)

        # 4. Run 3-way Hardware Comparison (CPU vs GPU vs NPU)
        print("\n[pipeline] Running 3-way hardware comparison across models...")
        for model in self.config.models:
            self.runner.run([
                sys.executable, "analysis/hw_comparison.py",
                "--model", str(model),
                "--iterations", str(self.config.iterations),
                "--repeats", str(self.config.repeats)
            ])

        # 5. Finalize figures
        copied = FigureCollector.copy_results_pngs(self.config.results_dir, self.config.figures_dir)
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"[pipeline] Copied {len(copied)} figure(s) to: {self.config.figures_dir}")
        print(f"[{end_time.strftime('%H:%M:%S')}] End-to-end flow complete.")
        print(f"Total pipeline duration: {duration:.2f} seconds.")


def parse_args() -> PipelineConfig:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the integrated benchmark pipeline.")
    parser.add_argument("--models-dir", default=str(project_root / "models"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--results-dir", default=str(project_root / "results"))
    parser.add_argument("--figures-dir", default=str(project_root / "paper" / "figures"))

    args = parser.parse_args()
    models = sorted(Path(args.models_dir).resolve().glob("*.onnx"))
    
    if not models:
        raise ValueError("No models found.")

    return PipelineConfig(
        project_root=project_root,
        models=models,
        results_dir=Path(args.results_dir).resolve(),
        figures_dir=Path(args.figures_dir).resolve(),
        iterations=args.iterations,
        repeats=args.repeats
    )


def main() -> None:
    config = parse_args()
    PipelineRunner(config).run()


if __name__ == "__main__":
    main()
