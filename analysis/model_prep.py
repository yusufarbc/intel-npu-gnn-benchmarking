import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    APPNP,
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    SGConv,
    TransformerConv,
)

import nncf

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ImportError:
    quantize_dynamic = None
    QuantType = None
    print("onnxruntime not found. Please pip install onnxruntime")

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K=K)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class APPNPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.prop1 = APPNP(K, alpha)

    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, beta=False)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, beta=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"File already exists: {dest.name}")
        return
    print(f"Downloading {url} to {dest}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    except Exception as exc:
        print(f"Download failed for {dest.name}: {exc}")

def quantize_model(input_path: Path, output_path: Path):
    if output_path.exists():
        print(f"Quantized model already exists: {output_path.name}")
        return
    print(f"Quantizing {input_path.name} to INT8...")
    try:
        if quantize_dynamic is None:
            raise RuntimeError("onnxruntime quantization not available")
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
        )
        print(f"Quantization complete: {output_path.name}")
    except Exception as e:
        print(f"Failed to quantize {input_path.name}: {e}")

def export_gnn_models(models_dir: Path) -> None:
    # Cora dataset size: 2708 nodes, 1433 features, 7 classes
    num_nodes = 2708
    num_features = 1433
    num_classes = 7
    edge_index = torch.randint(0, num_nodes, (2, 10000))
    x = torch.randn(num_nodes, num_features)

    model_configs = [
        ("GCN", GCN(num_features, 64, num_classes)),
        ("GraphSAGE", GraphSAGE(num_features, 64, num_classes)),
        ("GAT", GAT(num_features, 32, num_classes, heads=1)),
        ("GIN", GIN(num_features, 64, num_classes)),
        ("SGC", SGC(num_features, num_classes)),
        ("APPNP", APPNPModel(num_features, 64, num_classes)),
        ("GraphTransformer", GraphTransformer(num_features, 32, num_classes, heads=1)),
    ]

    for name, model in model_configs:
        model.eval()
        output_path = models_dir / f"{name}_fp32.onnx"

        print(f"Exporting {name} to {output_path}...")
        use_constant_folding = name not in ["GAT", "GraphTransformer"]

        torch.onnx.export(
            model,
            (x, edge_index),
            str(output_path),
            input_names=["x", "edge_index"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=use_constant_folding,
        )
        print("Done.")

        try:
            quantize_gnn_to_int8(output_path)
        except Exception as e:
            print(f"Quantization failed for {name}: {e}")

    standardize_existing_models(models_dir)
    for f in models_dir.glob("*_fp32.onnx"):
        int8_name = f.parent / f"{f.name.replace('_fp32', '_int8')}"
        if not int8_name.exists():
            try:
                quantize_gnn_to_int8(f)
            except Exception as e:
                print(f"Failed to quantize {f.name}: {e}")


def standardize_existing_models(models_dir: Path) -> None:
    import shutil

    for f in models_dir.glob("*.onnx"):
        if "_fp32" not in f.name and "_int8" not in f.name:
            new_name = f.parent / f"{f.stem}_fp32.onnx"
            if not new_name.exists():
                shutil.move(str(f), str(new_name))
                print(f"Renamed {f.name} -> {new_name.name}")
            else:
                f.unlink()


def quantize_gnn_to_int8(onnx_path: Path) -> None:
    import onnx

    int8_path = onnx_path.parent / f"{onnx_path.name.replace('_fp32', '_int8')}"
    if int8_path.exists():
        print(f"{int8_path.name} already exists, skipping quantization.")
        return

    print(f"Quantizing {onnx_path.name} to INT8...")
    model = onnx.load(str(onnx_path))

    def transform_fn(data_item):
        return data_item

    calibration_data = []
    input_info = []
    for input_node in model.graph.input:
        name = input_node.name
        shape = []
        for dim in input_node.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                if any(x in name.lower() for x in ["node", "edge", "x", "input", "data"]):
                    if "transformer" in onnx_path.name.lower():
                        if "node" in name.lower() or name.lower() == "x":
                            shape.append(2708)
                        elif "edge" in name.lower():
                            shape.append(10000)
                        else:
                            shape.append(1)
                    else:
                        shape.append(2708 if "node" in name or name == "x" else 10000)
                else:
                    shape.append(1)

        shape = [s if isinstance(s, int) and s > 0 else 1 for s in shape]

        dtype = np.float32
        if input_node.type.tensor_type.elem_type == 7:
            dtype = np.int64

        input_info.append((name, tuple(shape), dtype))

    num_calibration_samples = 50
    for _ in range(num_calibration_samples):
        item = {}
        for name, shape, dtype in input_info:
            if dtype == np.float32:
                item[name] = (np.random.randn(*shape).astype(dtype) * 0.1)
            else:
                max_value = 2708 if "edge" in name.lower() or "index" in name.lower() else 100
                item[name] = np.random.randint(0, max_value, shape).astype(dtype)
        calibration_data.append(item)

    calibration_dataset = nncf.Dataset(calibration_data, transform_fn)
    quantized_model = nncf.quantize(model, calibration_dataset)
    onnx.save(quantized_model, str(int8_path))
    print(f"Saved INT8 model to {int8_path}")


def prepare_baseline_models(models_dir: Path) -> None:
    models = {
        "resnet50": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx",
        "mobilenetv2": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
    }
    bert_tiny_url = "https://huggingface.co/optimum/bert-tiny-uncased/resolve/main/model.onnx"

    print("Fetching Vision Models (FP32)...")
    for name, url in models.items():
        fp32_path = models_dir / f"{name}_fp32.onnx"
        download_file(url, fp32_path)

    print("\nProcessing CNN INT8 variants...")
    for name in models.keys():
        fp32_path = models_dir / f"{name}_fp32.onnx"
        int8_path = models_dir / f"{name}_int8.onnx"
        if fp32_path.exists():
            quantize_model(fp32_path, int8_path)

    print("\nProcessing NLP/Transformer Models...")
    bert_fp32 = models_dir / "bert-tiny_fp32.onnx"
    if not bert_fp32.exists():
        download_file(bert_tiny_url, bert_fp32)

    if bert_fp32.exists():
        bert_int8 = models_dir / "bert-tiny_int8.onnx"
        quantize_model(bert_fp32, bert_int8)


def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    export_gnn_models(models_dir)
    prepare_baseline_models(models_dir)

    print("\nAcademic Model Preparation Complete.")
    print("You now have GNNs plus CNN/NLP baselines ready.")

if __name__ == "__main__":
    main()
