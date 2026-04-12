import argparse
import urllib.request
from pathlib import Path
import os

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("onnxruntime not found. Please pip install onnxruntime")

def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"File already exists: {dest.name}")
        return
    print(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")

def quantize_model(input_path: Path, output_path: Path):
    if output_path.exists():
        print(f"Quantized model already exists: {output_path.name}")
        return
    print(f"Quantizing {input_path.name} to INT8...")
    try:
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
        )
        print(f"Quantization complete: {output_path.name}")
    except Exception as e:
        print(f"Failed to quantize {input_path.name}: {e}")

def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Diverse Academic Workloads
    models = {
        "resnet50": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx",
        "mobilenetv2": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    }

    print("Fetching Vision Models (FP32)...")
    for name, url in models.items():
        fp32_path = models_dir / f"{name}_fp32.onnx"
        download_file(url, fp32_path)

    # Dynamic Quantization is optimal for MatMul/Attention (NLP), but causes cyclical 
    # graph errors in older ONNX CV models due to unsupported Conv quantization.
    print("\nProcessing NLP/Transformer Models...")
    bert_fp32 = models_dir / "BERT-tiny.onnx"
    if bert_fp32.exists():
        bert_int8 = models_dir / "BERT-tiny_int8.onnx"
        quantize_model(bert_fp32, bert_int8)

    print("\nAcademic Model Preparation Complete.")
    print("You now have a diverse mix of workloads: CNNs (Vision) and Transformers (NLP with INT8 variant).")

if __name__ == "__main__":
    main()
