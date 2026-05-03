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

    # Expect a hw_comparison output directory per active model stem
    expected_hw = sorted(p.stem.lower() for p in active_models)
    found_hw = {p.name.lower() for p in hw_dir.iterdir() if p.is_dir()} if hw_dir.exists() else set()
    missing_hw = sorted(name for name in expected_hw if name not in found_hw)
    if missing_hw:
        print("[FAIL] Missing hw_comparison outputs:", ", ".join(missing_hw))
    else:
        print("[OK] hw_comparison outputs complete.")

    matrix_path = results_dir / "scalability_matrix.csv"
    if not matrix_path.exists():
        print("[FAIL] Missing scalability_matrix.csv (run Step 2).")
    else:
        df = pd.read_csv(matrix_path)
        print(f"[OK] scalability_matrix.csv present ({len(df)} rows).")
        if "model" in df.columns:
            matrix_models = {str(m).lower() for m in df["model"].dropna().tolist()}
            expected_models = {p.stem.lower() for p in active_models}
            missing_models = sorted(expected_models - matrix_models)
            if missing_models:
                print("[FAIL] Missing models in scalability_matrix.csv:", ", ".join(missing_models))
            else:
                print("[OK] scalability_matrix.csv covers all active models.")

if __name__ == "__main__":
    main()
