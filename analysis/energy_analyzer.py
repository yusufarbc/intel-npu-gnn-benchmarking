import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class EnergyAnalyzer:
    """Parses Intel SoC Watch power logs or estimates NPU power consumption.

    Supported log formats:
    - *_trace.csv  produced by  socwatch -r int  (Power in mJ column)
    - *.csv        produced by  socwatch -r sum  (summary, fallback)
    """

    def __init__(self, log_path: Optional[Path] = None, sample_period_s: float = 1.0):
        self.log_path = log_path
        self.sample_period_s = sample_period_s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv_robust(path: Path) -> pd.DataFrame:
        """Try multiple encodings; skip malformed lines."""
        for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc, sep=",", on_bad_lines="skip")
            except Exception:
                pass
        raise RuntimeError(f"Cannot read {path} with any supported encoding")

    @staticmethod
    def _pick_column(columns: pd.Index, keywords: list[str]) -> Optional[str]:
        scored: list[Tuple[int, str]] = []
        for col in columns:
            name = str(col).lower()
            score = sum(
                (len(keywords) - i) * 10
                for i, kw in enumerate(keywords)
                if kw in name
            )
            if score:
                scored.append((score, col))
        scored.sort(reverse=True)
        return scored[0][1] if scored else None

    @classmethod
    def _pick_power_column(cls, columns: pd.Index) -> Optional[str]:
        return cls._pick_column(
            columns,
            ["npu", "ai boost", "ai", "soc", "package", "pkg", "power"],
        )

    @classmethod
    def _pick_time_column(cls, columns: pd.Index) -> Optional[str]:
        return cls._pick_column(
            columns,
            ["continuous time", "timestamp", "time", "elapsed", "seconds", "sec"],
        )

    @staticmethod
    def _parse_time_series(series: pd.Series) -> Optional[pd.Series]:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().any():
            return (parsed - parsed.iloc[0]).dt.total_seconds()
        return None

    # ------------------------------------------------------------------
    # Trace CSV parser  (*_trace.csv from  socwatch -r int)
    # ------------------------------------------------------------------

    def _parse_trace_csv(self, path: Path) -> Optional[Dict[str, Any]]:
        """Parse socwatch -r int trace CSV.

        The file looks like:
            ...header lines...
            Package Power - CPU/Package_0
            Sample #, Continuous Time (usec), Duration (ms), Power (mJ)
            1, 12345, 100.0, 512.3
            ...
        We find the data block by locating the header row.
        """
        try:
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            return None

        lines = raw.splitlines()

        # Find the CSV header row (contains 'Sample' and 'Power')
        header_idx = None
        for i, ln in enumerate(lines):
            ll = ln.lower()
            if "sample" in ll and "power" in ll:
                header_idx = i
                break

        if header_idx is None:
            return None

        import io
        block = "\n".join(lines[header_idx:])
        try:
            df = pd.read_csv(io.StringIO(block), on_bad_lines="skip")
        except Exception:
            return None

        # Find power column (values are in mJ -> convert to W using duration)
        power_col = self._pick_power_column(df.columns)
        time_col  = self._pick_time_column(df.columns)
        dur_col   = next(
            (c for c in df.columns if "duration" in str(c).lower()), None
        )

        if power_col is None:
            return None

        power_mj = pd.to_numeric(df[power_col], errors="coerce").dropna()
        if power_mj.empty:
            return None

        # If we have duration, compute average power in Watts per sample
        if dur_col is not None:
            dur_ms = pd.to_numeric(df[dur_col], errors="coerce")
            valid  = pd.concat([power_mj, dur_ms], axis=1).dropna()
            if not valid.empty:
                # P(W) = E(mJ) / t(ms)  (units cancel correctly)
                valid.columns = ["e_mj", "d_ms"]
                valid = valid[valid["d_ms"] > 0]
                if not valid.empty:
                    power_w_series = valid["e_mj"] / valid["d_ms"]  # mJ/ms = W
                    avg_power_w = float(power_w_series.mean())
                    total_energy_j = float((valid["e_mj"] / 1000.0).sum())
                    duration_s = float(valid["d_ms"].sum() / 1000.0)
                    return {
                        "avg_npu_power_w": avg_power_w,
                        "avg_power_w":     avg_power_w,
                        "max_npu_power_w": float(power_w_series.max()),
                        "min_npu_power_w": float(power_w_series.min()),
                        "sample_count":    int(len(valid)),
                        "duration_s":      duration_s,
                        "energy_j":        total_energy_j,
                        "power_column":    str(power_col),
                        "source":          "socwatch_trace",
                    }

        # Fallback: treat mJ values as if each sample is sample_period_s long
        avg_power_w = float(power_mj.mean() / (self.sample_period_s * 1000))
        energy_j    = float(power_mj.sum() / 1000.0)
        return {
            "avg_npu_power_w": avg_power_w,
            "avg_power_w":     avg_power_w,
            "sample_count":    int(len(power_mj)),
            "energy_j":        energy_j,
            "source":          "socwatch_trace_fallback",
        }

    # ------------------------------------------------------------------
    # Summary CSV parser  (*.csv from  socwatch -r sum)
    # ------------------------------------------------------------------

    def _parse_summary_csv(
        self,
        path: Path,
        start_time: Optional[str],
        end_time: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Parse the socwatch summary CSV.

        The summary CSV has a multi-line header block followed by a table
        that looks like:
            Metric, Average, ...
            Package Power, 8.23 W, ...
        We skip header lines and look for a row with 'Power' in the first col.
        """
        try:
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            return None

        lines = raw.splitlines()

        # Strategy 1: find a data row with "Average Rate" column
        # Strategy 2: look for "Package Power" text value row
        # The summary CSV contains:  Metric, Average Rate (W), Total (J), ...
        # Find header line
        header_idx = None
        for i, ln in enumerate(lines):
            if "average" in ln.lower() and "metric" in ln.lower():
                header_idx = i
                break
            if "average rate" in ln.lower():
                header_idx = i
                break

        if header_idx is not None:
            import io
            block = "\n".join(lines[header_idx:])
            try:
                df = pd.read_csv(io.StringIO(block), on_bad_lines="skip")
                # Find package power row
                metric_col = df.columns[0]
                pwr_rows = df[df[metric_col].astype(str).str.lower().str.contains("package power|pkg.power", na=False)]
                if not pwr_rows.empty:
                    row = pwr_rows.iloc[0]
                    # Find numeric value columns
                    nums = pd.to_numeric(row, errors="coerce").dropna()
                    if not nums.empty:
                        avg_w = float(nums.iloc[0])
                        energy_j = float(nums.iloc[1]) if len(nums) > 1 else float("nan")
                        return {
                            "avg_npu_power_w": avg_w,
                            "avg_power_w":     avg_w,
                            "energy_j":        energy_j,
                            "source":          "socwatch_summary",
                        }
            except Exception:
                pass

        # Strategy 3: plain generic CSV attempt
        try:
            df = self._read_csv_robust(path)
            power_col = self._pick_power_column(df.columns)
            time_col  = self._pick_time_column(df.columns)
            if power_col:
                work_df = df
                if start_time and end_time and time_col:
                    s = work_df[time_col].astype(str)
                    mask = (s >= start_time[:8]) & (s <= end_time[:8])
                    if mask.any():
                        work_df = work_df.loc[mask]

                power_series = pd.to_numeric(work_df[power_col], errors="coerce").dropna()
                if not power_series.empty:
                    avg_power = float(power_series.mean())
                    duration_s = float(len(power_series) * self.sample_period_s)
                    energy_j   = float(avg_power * duration_s)
                    return {
                        "avg_npu_power_w": avg_power,
                        "avg_power_w":     avg_power,
                        "max_npu_power_w": float(power_series.max()),
                        "min_npu_power_w": float(power_series.min()),
                        "sample_count":    int(len(power_series)),
                        "duration_s":      duration_s,
                        "energy_j":        energy_j,
                        "source":          "socwatch_generic",
                    }
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        start_time: Optional[str] = None,
        end_time:   Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse the log file and return power/energy metrics."""
        if not self.log_path or not self.log_path.exists():
            return {"error": "Log file not found"}

        path = self.log_path

        # Prefer _trace.csv (more accurate, per-sample data)
        trace_path = path.parent / (path.stem + "_trace.csv")
        if trace_path.exists():
            result = self._parse_trace_csv(trace_path)
            if result:
                return result

        # Try the main CSV as trace format first, then summary
        result = self._parse_trace_csv(path)
        if result:
            return result

        result = self._parse_summary_csv(path, start_time, end_time)
        if result:
            return result

        return {"error": "No parseable power data found in log"}

    def estimate(self, latency_ms: float, tdp_w: float = 5.0) -> Dict[str, Any]:
        """Estimate energy from latency and typical NPU TDP."""
        energy_j = (tdp_w * latency_ms) / 1000.0
        return {
            "avg_npu_power_w":    tdp_w,
            "avg_power_w":        tdp_w,
            "energy_j":           energy_j,
            "estimated_energy_j": energy_j,
            "note":               "estimated",
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SoC Watch energy log analyzer")
    parser.add_argument("--log",    help="Path to SoC Watch CSV log (*.csv or *_trace.csv)")
    parser.add_argument("--matrix", help="Path to scalability_matrix.csv to enrich with energy data")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None
    analyzer = EnergyAnalyzer(log_path)

    if args.matrix:
        matrix_path = Path(args.matrix)
        df = pd.read_csv(matrix_path)
        results = []
        for _, row in df.iterrows():
            res = analyzer.analyze(
                str(row.get("start_time", "")),
                str(row.get("end_time",   "")),
            )
            if "error" in res:
                res = analyzer.estimate(float(row.get("o_mean_ms", 0) or 0))

            avg_pw  = float(res.get("avg_power_w", res.get("avg_npu_power_w", 0)) or 0)
            o_mean  = float(row.get("o_mean_ms", 0) or 0)
            e_j     = res.get("energy_j") or (avg_pw * o_mean / 1000.0)
            e_inf   = (avg_pw * o_mean / 1000.0) if avg_pw and o_mean else 0.0
            inf_j   = (1.0 / e_inf) if e_inf else 0.0
            results.append({
                "avg_npu_power_w":  avg_pw,
                "energy_j":         e_j,
                "energy_per_inf_j": e_inf,
                "inf_per_j":        inf_j,
            })

        edf = pd.DataFrame(results)
        for col in edf.columns:
            df[col] = edf[col]
        df.to_csv(matrix_path, index=False)
        print(f"Matrix updated: {matrix_path}")
    else:
        if log_path:
            print(analyzer.analyze())
        else:
            print("No log path provided. Use --log or --matrix.")
