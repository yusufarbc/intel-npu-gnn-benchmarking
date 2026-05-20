from pathlib import Path
import json

models_dir = Path('models')
models = list(models_dir.glob('*_fp32.onnx'))
results = []

print("Model Size Analysis (MB)")
print("=" * 70)
print(f"{'Model':<20} {'FP32 Total':<12} {'INT8':<12} {'Ratio':<8} {'Status':<15}")
print("-" * 70)

for fp32 in sorted(models):
    name = fp32.stem.replace('_fp32', '')
    int8 = fp32.parent / f'{name}_int8.onnx'
    fp32_data = fp32.parent / f'{name}_fp32.onnx.data'
    
    # Total FP32 size (onnx + data file)
    fp32_onnx_size = fp32.stat().st_size
    fp32_data_size = fp32_data.stat().st_size if fp32_data.exists() else 0
    fp32_total = fp32_onnx_size + fp32_data_size
    
    # INT8 size
    int8_size = int8.stat().st_size if int8.exists() else 0
    
    if int8_size > 0:
        ratio = fp32_total / int8_size if int8_size > 0 else 0
        
        # Determine status
        if int8_size > fp32_total:
            status = "WARNING: INT8 BIGGER!"
        elif ratio < 2:
            status = "Low compression"
        else:
            status = "OK"
        
        print(f"{name:<20} {fp32_total/1_048_576:>10.2f}   {int8_size/1_048_576:>10.2f}   {ratio:>6.2f}x   {status:<15}")

print("-" * 70)
print("\nNotes:")
print("- FP32 Total = .onnx file + .onnx.data file (external weights)")
print("- INT8 = embedded weights (no separate .data file)")
print("- Expected ratio: ~3-4x for good INT8 compression")
