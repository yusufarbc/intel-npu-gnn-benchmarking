import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, SGConv, APPNP, TransformerConv
from pathlib import Path
import nncf
import numpy as np

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
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, out_channels))
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
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def export_models():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Cora dataset size: 2708 nodes, 1433 features, 7 classes
    num_nodes = 2708
    num_features = 1433
    num_classes = 7
    edge_index = torch.randint(0, num_nodes, (2, 10000))
    x = torch.randn(num_nodes, num_features)

    model_configs = [
        ("GCN", GCN(num_features, 64, num_classes)),
        ("GraphSAGE", GraphSAGE(num_features, 64, num_classes)),
        ("GAT", GAT(num_features, 16, num_classes, heads=8)),
        ("GIN", GIN(num_features, 64, num_classes)),
        ("SGC", SGC(num_features, num_classes)),
        ("APPNP", APPNPModel(num_features, 64, num_classes)),
        ("GraphTransformer", GraphTransformer(num_features, 32, num_classes, heads=4)),
    ]

    for name, model in model_configs:
        model.eval()
        output_path = models_dir / f"{name}_fp32.onnx"
        print(f"Exporting {name} to {output_path}...")
        
        # We use a wrapper to handle the tuple of (x, edge_index)
        # Use legacy exporter (dynamo=False) for better GNN support
        torch.onnx.export(
            model,
            (x, edge_index),
            str(output_path),
            input_names=["x", "edge_index"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False
        )
        print(f"Done.")
        
        # Quantize to INT8
        try:
            quantize_to_int8(output_path)
        except Exception as e:
            print(f"Quantization failed for {name}: {e}")

    # Standardize names first
    standardize_existing_models(models_dir)
    
    # Ensure every _fp32 model has an _int8 version
    for f in models_dir.glob("*_fp32.onnx"):
        int8_name = f.parent / f"{f.name.replace('_fp32', '_int8')}"
        if not int8_name.exists():
            try:
                quantize_to_int8(f)
            except Exception as e:
                print(f"Failed to quantize {f.name}: {e}")

def standardize_existing_models(models_dir: Path):
    import shutil
    for f in models_dir.glob("*.onnx"):
        if "_fp32" not in f.name and "_int8" not in f.name:
            new_name = f.parent / f"{f.stem}_fp32.onnx"
            if not new_name.exists():
                shutil.move(str(f), str(new_name))
                print(f"Renamed {f.name} -> {new_name.name}")
            else:
                f.unlink() # Delete duplicate

def quantize_to_int8(onnx_path: Path):
    import onnx
    from onnx import version_converter
    print(f"Quantizing {onnx_path.name} to INT8...")
    model = onnx.load(str(onnx_path))
    
    # Force Opset upgrade to 13 (required for quantization)
    if model.opset_import[0].version < 13:
        print(f"  -> Upconverting {onnx_path.name} from opset {model.opset_import[0].version} to 13...")
        model.opset_import[0].version = 13
        # Ensure we save it to refresh internal metadata
        onnx.save(model, str(onnx_path))
        model = onnx.load(str(onnx_path))
    
    # Create a dummy calibration dataset
    # We need to match the input names and shapes
    def transform_fn(data_item):
        return data_item

    # Detect input names and shapes
    calibration_data = []
    input_info = []
    for input_node in model.graph.input:
        name = input_node.name
        shape = []
        for dim in input_node.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                # Handle dynamic dims (common in GNNs and Transformers)
                if any(x in name.lower() for x in ["node", "edge", "x", "input", "data"]):
                    # Heuristic for different model types
                    if "bert" in onnx_path.name.lower():
                        shape.append(1 if len(shape) == 0 else (128 if len(shape) == 1 else 1)) # [1, 128]
                    else:
                        shape.append(2708 if "node" in name or "x" == name else 10000)
                else:
                    shape.append(1)
        
        # Determine data type
        dtype = np.float32
        if input_node.type.tensor_type.elem_type == 7: # INT64
            dtype = np.int64
        
        input_info.append((name, tuple(shape), dtype))

    for _ in range(5):
        item = {}
        for name, shape, dtype in input_info:
            if dtype == np.float32:
                item[name] = np.random.randn(*shape).astype(dtype)
            else:
                item[name] = np.random.randint(0, 100, shape).astype(dtype)
        calibration_data.append(item)

    calibration_dataset = nncf.Dataset(calibration_data, transform_fn)
    
    quantized_model = nncf.quantize(model, calibration_dataset)
    
    int8_path = onnx_path.parent / f"{onnx_path.name.replace('_fp32', '_int8')}"
    onnx.save(quantized_model, str(int8_path))
    print(f"Saved INT8 model to {int8_path}")

if __name__ == "__main__":
    export_models()
