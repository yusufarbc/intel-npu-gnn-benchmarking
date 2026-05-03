import sys
from pathlib import Path
import pandas as pd

# Setup path so we can import from analysis folder
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.energy_analyzer import EnergyAnalyzer

def main():
    RESULTS_DIR = Path("results")
    MATRIX_FILE = RESULTS_DIR / "scalability_matrix.csv"

    if not MATRIX_FILE.exists():
        print("scalability_matrix.csv not found. Run Step 2 first.")
    else:
        df_m = pd.read_csv(MATRIX_FILE)

        energy_filled = (
            "avg_npu_power_w" in df_m.columns
            and df_m["avg_npu_power_w"].notna().all()
        )

        if energy_filled:
            print("Energy data fully populated in scalability_matrix.csv.")
            print(df_m[["model", "avg_npu_power_w", "energy_j", "inf_per_j"]].to_string(index=False))
        else:
            print("Energy columns have missing data. Trying to populate from ETL/CSV logs...")

            # 1. Look for already-converted CSV power logs
            log_csvs = sorted(RESULTS_DIR.glob("energy_log_hw_comp_*power*.csv"))

            # 2. If none, we can't do much without ETL to CSV conversion here,
            #    which is handled elsewhere or by socwatch directly.
            #    The notebook had some `_convert_etl_to_csv` which isn't defined.
            #    We'll rely on the existing CSVs or fallback to estimation.

            if not log_csvs:
                print("  No log files available; using estimation mode.")
                for idx, row in df_m.iterrows():
                    o_mean = float(row.get("o_mean_ms", 0) or 0)
                    res    = EnergyAnalyzer().estimate(o_mean)
                    avg_pw = float(res.get("avg_power_w", 0))
                    e_inf  = (avg_pw * o_mean / 1000.0) if avg_pw and o_mean else float("nan")
                    df_m.at[idx, "avg_npu_power_w"]  = avg_pw
                    df_m.at[idx, "energy_j"]         = res.get("energy_j", float("nan"))
                    df_m.at[idx, "energy_per_inf_j"] = e_inf
                    df_m.at[idx, "inf_per_j"]        = (1.0 / e_inf) if e_inf else float("nan")
            else:
                # Map CSV filename -> model stem
                csv_map = {}
                for csv in log_csvs:
                    stem = csv.name.replace("energy_log_hw_comp_", "")
                    for suffix in ("_power.csv", ".csv"):
                        if stem.endswith(suffix):
                            stem = stem[: -len(suffix)]
                            break
                    csv_map[stem.lower()] = csv

                for idx, row in df_m.iterrows():
                    model_key   = str(row.get("model", "")).lower()
                    matched_csv = next(
                        (p for k, p in csv_map.items()
                         if k in model_key or model_key in k),
                        None
                    )
                    start_t = str(row.get("start_time", ""))
                    end_t   = str(row.get("end_time",   ""))
                    o_mean  = float(row.get("o_mean_ms", 0) or 0)

                    if matched_csv and Path(matched_csv).exists():
                        res = EnergyAnalyzer(Path(matched_csv)).analyze(start_t, end_t)
                    else:
                        res = EnergyAnalyzer().estimate(o_mean)

                    avg_pw = float(res.get("avg_power_w", res.get("avg_npu_power_w", 0)) or 0)
                    e_j    = res.get("energy_j") or (avg_pw * o_mean / 1000.0 if avg_pw else float("nan"))
                    e_inf  = (avg_pw * o_mean / 1000.0) if avg_pw and o_mean else float("nan")
                    df_m.at[idx, "avg_npu_power_w"]  = avg_pw
                    df_m.at[idx, "energy_j"]         = e_j
                    df_m.at[idx, "energy_per_inf_j"] = e_inf
                    df_m.at[idx, "inf_per_j"]        = (1.0 / e_inf) if e_inf else float("nan")

            df_m.to_csv(MATRIX_FILE, index=False)
            print("scalability_matrix.csv updated with energy data.")
            print(df_m[["model", "avg_npu_power_w", "energy_j", "inf_per_j"]].to_string(index=False))

if __name__ == "__main__":
    main()
