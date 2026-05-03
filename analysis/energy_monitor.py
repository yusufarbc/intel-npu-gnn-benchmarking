import os
import sys
import shutil
import time
import subprocess
import threading
import queue
import signal
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def _socwatch_available():
    return shutil.which("socwatch") is not None

def run_with_energy_monitor(command, session_name, enabled=True, warmup_s=2):
    """
    Intel SoC Watch power monitoring wrapper.

    Uses Popen so socwatch runs in the background while the benchmark
    executes in a parallel thread. On benchmark completion, sends
    CTRL_BREAK_EVENT so socwatch finalises its ETL files cleanly and
    produces the _trace.csv (per-sample power data).
    """
    output_base = RESULTS_DIR / f"energy_log_{session_name}"

    if not enabled:
        print(f"Running (monitor disabled): {command}")
        subprocess.run(command, shell=True, check=True)
        return None

    if not _socwatch_available():
        print("WARNING: SoC Watch not found in PATH. Running without energy monitoring.")
        subprocess.run(command, shell=True, check=True)
        return None

    # ---- Start SoC Watch in background ---------------------------------------
    sw_proc_holder = [None]

    def _run_sw():
        # Pre-cleanup: kill any zombie socwatch processes to free the driver
        try:
            subprocess.run(["taskkill", "/F", "/IM", "socwatch.exe", "/T"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass

        sw_cmd = [
            "socwatch",
            "-f", "power",
            "-r", "int",
            "-o", str(output_base),
        ]
        try:
            proc = subprocess.Popen(
                sw_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                text=True
            )
            sw_proc_holder[0] = proc
            stdout, stderr = proc.communicate()
            if proc.returncode != 0 and proc.returncode != 255: # 255 usually happens on CTRL_BREAK
                print(f"[ENERGY] SoC Watch Warning (Code {proc.returncode}):\n{stderr.strip() or stdout.strip()}")
        except Exception as exc:
            print(f"[ENERGY] SoC Watch process error: {exc}")

    sw_thread = threading.Thread(target=_run_sw, daemon=True)
    sw_thread.start()
    
    # Give SoC Watch time to initialize ETL sessions before benchmark starts
    time.sleep(warmup_s + 3)

    # ---- Run benchmark in a thread, measure wall-clock time ------------------
    bm_result = queue.Queue()

    def _run_bm():
        start = time.perf_counter()
        try:
            subprocess.run(command, shell=True, check=True)
            bm_result.put(("ok", time.perf_counter() - start))
        except Exception as exc:
            bm_result.put(("err", exc))

    bm_thread = threading.Thread(target=_run_bm, daemon=True)
    bm_thread.start()

    # ---- Wait for benchmark, then gracefully stop socwatch -------------------
    bm_thread.join()
    status, payload = bm_result.get()

    sw_proc = sw_proc_holder[0]
    if sw_proc and sw_proc.poll() is None:
        try:
            os.kill(sw_proc.pid, signal.CTRL_BREAK_EVENT)
        except Exception:
            pass
        try:
            sw_proc.wait(timeout=20)
        except Exception:
            sw_proc.kill()
    sw_thread.join(timeout=5)

    if status == "err":
        raise payload

    elapsed = payload
    print(f"[ENERGY] Benchmark done in {elapsed:.1f}s")

    # ---- Find output files ---------------------------------------------------
    trace_csv   = output_base.parent / (output_base.name + "_trace.csv")
    summary_csv = output_base.parent / (output_base.name + ".csv")

    if trace_csv.exists():
        print(f"[ENERGY] Trace CSV: {trace_csv.name}  ({trace_csv.stat().st_size // 1024} KB)")
        return trace_csv
    if summary_csv.exists():
        print(f"[ENERGY] Summary CSV: {summary_csv.name}")
        return summary_csv

    etls = sorted(RESULTS_DIR.glob(f"{output_base.name}*Session.etl"))
    if etls:
        print("[ENERGY] ETL files produced (no CSV yet):")
        for f in etls:
            print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
    else:
        print("WARNING [ENERGY] No output files found.")
    return None
