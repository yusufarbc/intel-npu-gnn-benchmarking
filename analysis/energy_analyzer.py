import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

class EnergyAnalyzer:
    """Parses HWiNFO CSV logs or provides estimates for NPU Power consumption."""
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path

    def analyze(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        if not self.log_path or not self.log_path.exists():
            return {"error": "Log file not found"}

        try:
            # HWiNFO CSVs often have multiple headers or specific encodings
            df = pd.read_csv(self.log_path, encoding='utf-16', sep=',', on_bad_lines='skip')
            
            # Find column related to NPU Power and Time
            power_cols = [c for c in df.columns if "NPU" in c and "Power" in c]
            time_cols = [c for c in df.columns if "Time" in c]
            
            if not power_cols:
                return {"error": "No NPU Power column found in HWiNFO log"}

            # Filter by time if provided
            if start_time and end_time and time_cols:
                t_col = time_cols[0]
                # Filter logic (assuming HH:MM:SS format)
                mask = (df[t_col] >= start_time) & (df[t_col] <= end_time)
                filtered_df = df.loc[mask]
                if filtered_df.empty:
                    # Try a more relaxed match if exact fails (e.g. ignoring seconds/milliseconds)
                    mask = (df[t_col].str[:8] >= start_time[:8]) & (df[t_col].str[:8] <= end_time[:8])
                    filtered_df = df.loc[mask]
                
                npu_power = filtered_df[power_cols[0]]
            else:
                npu_power = df[power_cols[0]]
            
            if npu_power.empty:
                return {"avg_npu_power_w": 0.0, "max_npu_power_w": 0.0, "sample_count": 0}

            return {
                "avg_npu_power_w": float(npu_power.mean()),
                "max_npu_power_w": float(npu_power.max()),
                "min_npu_power_w": float(npu_power.min()),
                "sample_count": int(len(npu_power))
            }
        except Exception as e:
            return {"error": f"Failed to parse energy log: {e}"}

    def estimate(self, latency_ms: float, tdp_w: float = 5.0) -> Dict[str, Any]:
        """Estimates energy consumption based on latency and typical NPU TDP."""
        # E = P * t
        energy_j = (tdp_w * latency_ms) / 1000.0
        return {
            "avg_npu_power_w": tdp_w,
            "estimated_energy_j": energy_j,
            "note": "estimated"
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", help="Path to HWiNFO CSV log")
    parser.add_argument("--matrix", help="Path to scalability_matrix.csv to enrich with energy data")
    args = parser.parse_args()
    
    log_path = Path(args.log) if args.log else None
    analyzer = EnergyAnalyzer(log_path)
    
    if args.matrix:
        matrix_path = Path(args.matrix)
        df = pd.read_csv(matrix_path)
        energy_results = []
        for _, row in df.iterrows():
            result = analyzer.analyze(row.get("start_time"), row.get("end_time"))
            if "error" in result:
                # Use estimation fallback if log is missing
                o_mean = row.get("o_mean_ms", 0.0)
                result = analyzer.estimate(o_mean)
                
            energy_results.append(result.get("avg_npu_power_w", 0.0))
        
        df["avg_npu_power_w"] = energy_results
        df.to_csv(matrix_path, index=False)
        print(f"✅ Matrix updated with energy data (fallback used if logs missing): {matrix_path}")
    else:
        if log_path:
            print(analyzer.analyze())
        else:
            print("No log path provided. Use --matrix to apply estimations.")
