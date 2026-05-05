from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.plot_config import apply_ieee_style, savefig_ieee, IEEE_COLORS

apply_ieee_style()


@dataclass
class LatencyModelConfig:
    results_dir: Path
    matrix_csv: Path


def _find_input_metadata(results_dir: Path, model_stem: str) -> Optional[Path]:
    # Typical structure: results/<model>/run_00/input_metadata.json
    model_dir = results_dir / model_stem
    if not model_dir.exists():
        return None

    # Prefer run_00 but accept any run.
    candidate = model_dir / "run_00" / "input_metadata.json"
    if candidate.exists():
        return candidate

    for run in sorted(model_dir.glob("run_*/input_metadata.json")):
        if run.exists():
            return run

    return None


def _load_metadata(results_dir: Path, model_stem: str) -> Dict[str, float]:
    path = _find_input_metadata(results_dir, model_stem)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: Dict[str, float] = {}
    for k in ("used_num_nodes", "used_num_edges", "used_num_features"):
        v = payload.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _fit_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    # Ordinary least squares.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def run_latency_model(config: LatencyModelConfig) -> Path:
    df = pd.read_csv(config.matrix_csv)
    if df.empty:
        raise RuntimeError(f"Empty matrix CSV: {config.matrix_csv}")

    # Attach dataset/graph metadata when available.
    meta_rows: List[Dict[str, float]] = []
    for m in df["model"].astype(str).tolist():
        meta = _load_metadata(config.results_dir, m)
        meta_rows.append(meta)

    meta_df = pd.DataFrame(meta_rows)
    df = pd.concat([df, meta_df], axis=1)

    # Select features (use the most explanatory ones available).
    feature_cols: List[str] = []
    for col in ("used_num_edges", "used_num_nodes", "params_mil", "ai"):
        if col in df.columns and df[col].notna().any():
            feature_cols.append(col)

    if not feature_cols:
        raise RuntimeError(
            "No usable feature columns found. Run scalability with --input-source auto to generate input_metadata.json."
        )

    y = df["o_mean_ms"].astype(float).to_numpy()
    X_raw = df[feature_cols].fillna(0.0).astype(float).to_numpy()

    # Add intercept.
    X = np.concatenate([np.ones((X_raw.shape[0], 1)), X_raw], axis=1)

    beta, r2 = _fit_linear_regression(X, y)
    yhat = X @ beta

    out_dir = config.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coeffs = {
        "target": "o_mean_ms",
        "features": ["intercept"] + feature_cols,
        "beta": [float(b) for b in beta],
        "r2": float(r2),
    }
    (out_dir / "latency_model_coeffs.json").write_text(json.dumps(coeffs, indent=2), encoding="utf-8")

    # Plot predicted vs actual.
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.scatter(y, yhat, s=18, color=IEEE_COLORS[0], edgecolor="k", linewidth=0.3, alpha=0.9)
    lim = [min(float(np.min(y)), float(np.min(yhat))), max(float(np.max(y)), float(np.max(yhat)))]
    ax.plot(lim, lim, "--", color="gray", linewidth=1)
    ax.set_xlabel("Measured latency (ms)")
    ax.set_ylabel("Predicted latency (ms)")
    ax.set_title(f"Simple latency model (R²={r2:.2f})")
    ax.grid(True, alpha=0.2)

    out_path = out_dir / "latency_model_pred_vs_actual"
    savefig_ieee(fig, out_path)
    plt.close(fig)

    return out_path.with_suffix(".png")


def parse_args() -> LatencyModelConfig:
    parser = argparse.ArgumentParser(description="Fit a simple latency model from benchmark results.")
    parser.add_argument("--results-dir", default="results", help="Results directory.")
    parser.add_argument(
        "--matrix",
        default=None,
        help="Path to scalability_matrix.csv (default: results/scalability_matrix.csv).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    matrix_csv = Path(args.matrix).resolve() if args.matrix else (results_dir / "scalability_matrix.csv")
    return LatencyModelConfig(results_dir=results_dir, matrix_csv=matrix_csv)


def main() -> None:
    config = parse_args()
    out = run_latency_model(config)
    print(f"Latency model saved: {out}")


if __name__ == "__main__":
    main()
