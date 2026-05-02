# Teknik Rehber: Python Analiz Scriptleri

Proje kapsamında geliştirilen analiz ve otomasyon scriptlerinin teknik rolleri ve kullanım detayları aşağıdadır.

## 1. Çekirdek İşlem Scriptleri

### `analysis/benchmark_runner.py`
Modelleri hedeflenen donanımda (NPU/GPU/CPU) koşturan ana motor.
*   **GNN Girdileri:** Cora veri seti yapısına uygun `x` ve `edge_index` tensörlerini üretir.
*   **Profilleme:** ONNX Runtime profiling izlerini (trace) JSON formatında oluşturur.

### `scripts/generate_gnn_models.py`
PyTorch Geometric (PyG) kullanarak 7 farklı GNN mimarisini ONNX formatına dönüştürür.
*   **Sabit Şekiller:** NPU uyumluluğu için dinamik shape'ler yerine statik shape padding uygular.

## 2. Analiz ve Görselleştirme Scriptleri

### `analysis/profiling_analyzer.py`
Donanım profil izlerini parse ederek akademik figürler üretir.
*   **Dışa Aktarma:** `latency_breakdown`, `fgr_diverging` ve `provider_fallback` grafiklerini 300 DPI ve SVG formatında üretir.
*   **Metrikler:** FGR ve CEI değerlerini hesaplayarak `advanced_metrics.json` dosyasına kaydeder.

### `analysis/scalability_analyzer.py`
Model boyutu ve karmaşıklığına göre performans değişimini ölçer.
*   **Roofline Modeli:** Donanım limitlerine göre model verimliliğini görselleştirir.
*   **Pareto Sınırı:** Parametre sayısı ve gecikme arasındaki dengeyi analiz eder.

### `analysis/hw_comparison.py`
Belirli bir model için CPU, iGPU ve NPU arasında 3 yönlü performans karşılaştırması yapar.

## 3. Otomasyon Hattı

### `run_pipeline.py`
Tüm süreci uçtan uca yöneten orkestratör. Modelleri üretir, tüm donanımlarda test eder, analizleri yapar ve sonuçları `paper/figures/` dizinine kopyalar.

---
*Son Güncelleme: 2 Mayıs 2026*
