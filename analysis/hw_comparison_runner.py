import sys
import glob
from pathlib import Path
import pandas as pd

# Setup path so we can import from analysis folder
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.energy_monitor import run_with_energy_monitor
from analysis.energy_analyzer import EnergyAnalyzer

def main():
    RUN_MODE = "full"   # "quick" | "full"
    AUTO_ENERGY_MONITOR = True

    RESULTS_DIR = Path("results")
    model_paths = [Path(p) for p in sorted(glob.glob("models/*.onnx")) 
                   if not str(p).endswith(".ort_broken.onnx") 
                   and "GAT" not in Path(p).name 
                   and "SGC" not in Path(p).name]

    if RUN_MODE == "quick":
        iterations, repeats = 10, 1
        allowlist = {"gcn_fp32.onnx", "graphsage_fp32.onnx", "resnet50_fp32.onnx"}
    else:
        iterations, repeats = 30, 1
        allowlist = None

    print(f"Found {len(model_paths)} models.")

    energy_csv_map = {}  # stem -> csv_path or None

    for model_path in model_paths:
        if allowlist and model_path.name.lower() not in allowlist:
            continue
        print(f"\nTarget: {model_path.name}")
        cmd = (
            f'"{sys.executable}" analysis/hw_comparison.py'
            f' --model "{model_path}"'
            f' --iterations {iterations}'
            f' --repeats {repeats}'
        )
        session = f"hw_comp_{model_path.stem}"
        try:
            csv = run_with_energy_monitor(cmd, session, enabled=AUTO_ENERGY_MONITOR)
            energy_csv_map[model_path.stem] = csv
        except Exception as e:
            print(f"  ❌ Benchmark failed for {model_path.name}: {e}")
            energy_csv_map[model_path.stem] = None

    # ---- Enrich scalability_matrix.csv with energy data ------------------------
    matrix_path = RESULTS_DIR / "scalability_matrix.csv"
    if matrix_path.exists() and energy_csv_map:
        df = pd.read_csv(matrix_path)
        
        # Pre-create columns if missing
        for col in ["avg_npu_power_w", "energy_j", "energy_per_inf_j", "inf_per_j"]:
            if col not in df.columns:
                df[col] = float("nan")

        # Optimization: Create a lookup map for faster matching
        for idx, row in df.iterrows():
            model_stem = str(row.get("model", "")).lower()
            if "sgc" in model_stem or "gat" in model_stem:
                continue

            # Find matching energy session
            match_stem = next((s for s in energy_csv_map if s.lower() in model_stem or model_stem in s.lower()), None)
            if not match_stem:
                continue

            csv_path = energy_csv_map[match_stem]
            o_mean = float(row.get("o_mean_ms", 0) or 0)

            try:
                if csv_path and Path(csv_path).exists():
                    res = EnergyAnalyzer(Path(csv_path)).analyze()
                else:
                    res = EnergyAnalyzer().estimate(o_mean)

                avg_pw = float(res.get("avg_power_w", res.get("avg_npu_power_w", 0)) or 0)
                e_inf  = (avg_pw * o_mean / 1000.0) if avg_pw and o_mean else float("nan")
                
                df.at[idx, "avg_npu_power_w"]  = avg_pw
                df.at[idx, "energy_per_inf_j"] = e_inf
                df.at[idx, "inf_per_j"]        = (1.0 / e_inf) if e_inf > 0 else float("nan")
            except Exception as e:
                print(f"  ⚠️ Energy correlation failed for {model_stem}: {e}")

        df.to_csv(matrix_path, index=False)
        print(f"\nscalability_matrix.csv updated with energy data.")
    else:
        print("scalability_matrix.csv not found or no energy data available.")

if __name__ == "__main__":
    main()
