# Benchmarking Workflow

NPU GNN Benchmarking pipeline'ını çalıştırmak için aşağıdaki adımları izleyin.

## Adım 1: Ortam Hazırlığı

Gerekli kütüphaneleri yükleyin:
```powershell
pip install torch torch-geometric torch-scatter torch-sparse onnx onnxruntime-openvino openvino numpy pandas matplotlib psutil onnxscript
```

## Adım 2: GNN Modellerinin Üretilmesi

Analiz edilecek GNN modellerini `models/` klasörüne ONNX olarak export etmek için:
```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/generate_gnn_models.py
```
Bu komut, Cora veri seti boyutlarında GCN, GAT, GraphSAGE gibi modelleri oluşturur.

## Adım 3: Tam Pipeline'ın Çalıştırılması

Tüm analizleri (profilleme, scalability, roofline ve donanım karşılaştırması) başlatmak için:
```powershell
python run_pipeline.py --iterations 100 --repeats 3 --profile-model models/GCN.onnx
```

### Parametreler:
*   `--iterations`: Ölçüm hassasiyeti için çıkarım sayısı.
*   `--repeats`: İstatistiksel güvenilirlik için her testin tekrar sayısı.
*   `--profile-model`: Detaylı operatör analizi (FGR/CEI) yapılacak ana model.

## Adım 4: Sonuçların İncelenmesi

Pipeline tamamlandığında şu çıktılar oluşur:
*   `results/advanced_metrics.json`: FGR ve CEI değerleri.
*   `results/latency_breakdown.png`: Gecikme bileşenleri grafiği.
*   `results/roofline_model.png`: Aritmetik yoğunluk analizi.
*   `paper/figures/`: Makalede kullanılmaya hazır tüm grafikler.
