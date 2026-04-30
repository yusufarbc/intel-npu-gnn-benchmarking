import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

class EnergyAnalyzer:
    """Parses HWiNFO CSV logs to extract NPU Power consumption."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def analyze(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        if not self.log_path.exists():
            return {"error": "Log file not found"}

        try:
            # HWiNFO CSVs often have multiple headers or specific encodings
            df = pd.read_csv(self.log_path, encoding='utf-16', sep=',', skipsearch=True, on_bad_lines='skip')
            
            # Find column related to NPU Power (names vary: "NPU Power", "NPU Rail Power", etc.)
            power_cols = [c for c in df.columns if "NPU" in c and "Power" in c]
            if not power_cols:
                return {"error": "No NPU Power column found in HWiNFO log"}

            npu_power = df[power_cols[0]]
            
            return {
                "avg_npu_power_w": npu_power.mean(),
                "max_npu_power_w": npu_power.max(),
                "min_npu_power_w": npu_power.min(),
                "sample_count": len(npu_power)
            }
        except Exception as e:
            return {"error": f"Failed to parse energy log: {e}"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to HWiNFO CSV log")
    args = parser.parse_args()
    
    analyzer = EnergyAnalyzer(Path(args.log))
    print(analyzer.analyze())
