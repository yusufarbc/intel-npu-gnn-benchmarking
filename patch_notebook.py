import nbformat as nbf
from pathlib import Path

notebook_path = 'npu_gnn_benchmarking.ipynb'
if not Path(notebook_path).exists():
    print(f"Error: {notebook_path} not found.")
    exit(1)

ntbk = nbf.read(notebook_path, as_version=4)

# 1. Update Initialization Cell (Cell 1)
init_code = """import os, sys, json, shutil, time, subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display, Markdown, HTML

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("paper/figures")

def run_with_energy_monitor(command, session_name, enabled=True):
    \"\"\"
    Intel SoC Watch background monitoring wrapper.
    Automates start/stop of power profiling during benchmark execution.
    \"\"\"
    if not enabled:
        print(f"🏃 Running (monitor disabled): {command}")
        subprocess.run(command, shell=True, check=True)
        return
        
    output_path = RESULTS_DIR / f"energy_log_{session_name}.csv"
    print(f"⚡ [ENERGY] Starting Intel SoC Watch monitor -> {output_path}")
    
    # Start SoC Watch in background (Requires Admin)
    try:
        # CREATE_NEW_PROCESS_GROUP = 0x00000010
        monitor_proc = subprocess.Popen(
            ["socwatch", "-f", "power", "-o", str(output_path), "-t", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=0x00000010 
        )
        
        time.sleep(2) # Stabilize sensors
        
        print(f"🚀 [BENCHMARK] Executing: {command}")
        subprocess.run(command, shell=True, check=True)
        
    finally:
        if 'monitor_proc' in locals():
            monitor_proc.terminate()
            print(f"✅ [ENERGY] Log saved: {output_path}")

plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="muted")
print("✅ Advanced Analytics Ready with Integrated Energy Monitoring.")"""

# Find the cell by ID or assume index 1 based on previous view
# We will iterate to find the correct cell by looking for common strings
for cell in ntbk.cells:
    if "RESULTS_DIR =" in cell.source and cell.cell_type == "code":
        cell.source = init_code
        print("Updated Initialization cell.")
        break

# 2. Update Hardware Comparison Loop (Step 4)
step4_code = """import glob
import sys
from pathlib import Path

# Performance Evaluation Configuration
RUN_MODE = "full" 
AUTO_ENERGY_MONITOR = True # Toggle automated SoC profiling

model_paths = [Path(p) for p in sorted(glob.glob("models/*_fp32.onnx"))]

if RUN_MODE == "quick":
    iterations, repeats = 10, 1
    allowlist = {"gcn_fp32.onnx", "graphsage_fp32.onnx", "resnet50_fp32.onnx"}
else:
    iterations, repeats = 30, 1
    allowlist = None

print(f"🔍 Found {len(model_paths)} models for evaluation.")

for model_path in model_paths:
    if allowlist and model_path.name.lower() not in allowlist:
        continue
        
    print(f"\\n🚀 Current Target: {model_path.name}")
    
    # Construct the benchmark command
    cmd = f"{sys.executable} analysis/hw_comparison.py --model {model_path} --iterations {iterations} --repeats {repeats}"
    
    # Run with energy monitoring wrapper
    session = f"hw_comp_{model_path.stem}"
    run_with_energy_monitor(cmd, session, enabled=AUTO_ENERGY_MONITOR)"""

# Find the hardware comparison cell
for cell in ntbk.cells:
    if "analysis/hw_comparison.py" in cell.source and "model_paths" in cell.source:
        cell.source = step4_code
        print("Updated Hardware Comparison cell.")
        break

# Save the updated notebook
nbf.write(ntbk, notebook_path)
print(f"Successfully patched {notebook_path}")
