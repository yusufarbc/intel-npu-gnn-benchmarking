from pathlib import Path
import pandas as pd
import onnx
from onnx import checker

def main():
    # Verify models and outputs (derived from what exists on disk)
    models_dir = Path("models")
    results_dir = Path("results")
    hw_dir = results_dir / "hw_comparison"

    active_models = sorted(p for p in models_dir.glob("*.onnx") if not p.name.endswith(".ort_broken.onnx"))
    if not active_models:
        print("[FAIL] No models found under models/. Run Step 1 first.")
    else:
        print(f"[OK] Active models found: {len(active_models)}")

    # Validate ONNX models (catch the common INT8-CNN invalid graph issue)
    bad = []
    for p in active_models:
        try:
            checker.check_model(onnx.load(str(p)))
        except Exception as e:
            bad.append((p.name, str(e)))
    if bad:
        print("[FAIL] Invalid ONNX models detected:")
        for name, msg in bad:
            print(f"  - {name}: {msg[:200]}")
    else:
        print("[OK] All active ONNX models pass onnx.checker.")

    # Expect a per-model output directory (new layout) or legacy hw_comparison outputs
    expected_hw = sorted(p.stem.lower() for p in active_models)
    found_flat = {p.name.lower() for p in results_dir.iterdir() if p.is_dir()}
    found_legacy = {p.name.lower() for p in hw_dir.iterdir() if p.is_dir()} if hw_dir.exists() else set()
    missing_hw = sorted(name for name in expected_hw if name not in found_flat and name not in found_legacy)
    if missing_hw:
        print("[FAIL] Missing per-model outputs:", ", ".join(missing_hw))
    else:
        print("[OK] Per-model outputs complete.")

    summary_path = results_dir / "benchmark_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print(f"[OK] benchmark_summary.csv present ({len(df)} rows).")
    else:
        matrices = list(results_dir.rglob("scalability_matrix.csv"))
        if not matrices:
            print("[FAIL] Missing scalability_matrix.csv outputs (run per-model cells).")
        else:
            print(f"[OK] Found {len(matrices)} scalability_matrix.csv files under results/.")

if __name__ == "__main__":
    main()
