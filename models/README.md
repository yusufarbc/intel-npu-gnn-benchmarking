# Models — Onnx Model Zoo

Pre-exported onnx models for benchmarking. Models span gnn, cnn, and transformer architectures in fp32 and int8 precisions.

## Model Inventory

### Graph Neural Networks (9 models)
| Model | Type | FP32 | INT8 | Params |
|-------|------|------|------|--------|
| GCN | Spectral | ✅ | ✅ | 0.10M |
| GAT | Attentional | ✅ | ❌ (.failed) | 0.11M |
| GATv2 | Attentional | ✅ | ❌ (.failed) | 0.13M |
| GIN | Expressive | ✅ | ✅ | 0.09M |
| GraphSAGE | Spatial | ✅ | ✅ | 0.18M |
| SGC | Simplified | ✅ | ✅ | 0.02M |
| APPNP | Propagation | ✅ | ✅ | 0.09M |
| GraphTransformer | Hybrid | ✅ | ✅ | 0.18M |
| MPNN | Message-Passing | ✅ | ✅ | 0.20M |

### Dense Baselines (5 models)
| Model | Type | FP32 | INT8 | Params |
|-------|------|------|------|--------|
| ResNet50 | CNN | ✅ | ✅ | 25.5M |
| MobileNetV2 | CNN | ✅ | ✅ | 3.5M |
| EfficientNet-B0 | CNN | ✅ | ✅ | 5.3M |
| ViT-Tiny | Vision Transformer | ✅ | ❌ | 5.7M |
| BERT-Tiny | NLP Transformer | ✅ | ✅ | 4.4M |

## Notes

- `.failed` files indicate INT8 quantization compilation failures (GAT, GATv2: unsupported scatter/gather).
- `.onnx.data` files are external weight files for large models.
- Models are excluded from git tracking (too large).
- To regenerate all models: `python analysis/model_prep.py`
