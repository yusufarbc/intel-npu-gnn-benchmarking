# Technical Guide: Python Scripts

Proje kapsamında geliştirilen ve güncellenen scriptlerin teknik detayları aşağıdadır.

## 1. Core Engine Scripts

### `analysis/benchmark_runner.py`
ONNX Runtime ve OpenVINO kullanarak modelleri hedeflenen donanımda (NPU/GPU/CPU) koşturur.
*   **GNN Desteği:** Node features (`x`) ve `edge_index` gibi GNN-spesifik girdileri otomatik üretir.
*   **Profilleme:** ORT profiling JSON dosyalarını oluşturur.
*   **Resource Tracking:** `psutil` ile CPU ve bellek kullanımını takip eder.

### `scripts/generate_gnn_models.py`
PyTorch Geometric kullanarak benchmark için standart GNN modellerini üretir.
*   **Desteklenen Modeller:** GCN, GraphSAGE, GAT, GIN, SGC, APPNP, GraphTransformer.
*   **ONNX Export:** Modelleri NPU uyumluluğu için legacy ONNX exporter (`dynamo=False`) ve sabit shape'lerle (Cora ölçeği) dışa aktarır.

## 2. Analysis Scripts

### `analysis/profiling_analyzer.py`
ORT tarafından üretilen JSON profillerini parse eder ve akademik raporları oluşturur.
*   **Gelişmiş Metrikler:** FGR ve CEI değerlerini hesaplar.
*   **Breakdown:** Toplam süreyi Compute, DMA ve Dispatch olarak ayrıştırır.
*   **Görselleştirme:** Operator bazlı hızlanma ve gecikme dağılım grafiklerini (`latency_breakdown.png`) üretir.

### `analysis/scalability_analyzer.py`
Farklı model boyutları ve mimarileri arasında performans karşılaştırması yapar.
*   **Roofline Visualization:** Her modelin aritmetik yoğunluğunu hesaplayıp Roofline grafiğine yerleştirir.
*   **Scalability Matrix:** Modellerin parametre sayısına göre performans değişimini `scalability_matrix.csv` dosyasına kaydeder.

## 3. Pipeline Orchestrator

### `run_pipeline.py`
Tüm süreci tek komutla yönetir.
*   Sırasıyla: Profilleme -> Analiz -> Scalability -> Hardware Comparison adımlarını çalıştırır.
*   Nihai görsel sonuçları `paper/figures/` klasörüne kopyalar.
