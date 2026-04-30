# npu-graph-opt-benchmarking

Bu çalışma, **Intel Core Ultra (Meteor Lake)** mimarisi üzerindeki NPU, GPU ve CPU performansını çok boyutlu olarak analiz eden bir framework'tür. Projenin temel odağı; Çizge Sinir Ağlarının (GNN) NPU üzerindeki performans sınırlarını, klasik CNN (ResNet) ve Transformer (BERT) modelleriyle kıyaslayarak (**Cross-Architectural Benchmarking**) ortaya koymak ve operatör füzyonu gibi grafik optimizasyonlarının bu heterojen iş yükleri üzerindeki etkilerini karakterize etmektir.

## Proje Yapısı

Proje, akademik pipeline standartlarına uygun olarak organize edilmiştir:

- `analysis/`: Profiler, ölçeklenebilirlik, enerji ve karmaşık metrik analiz araçları.
- `scripts/`: Model üretimi (`generate_gnn_models.py`) ve otomasyon scriptleri.
- `docs/`: Metodoloji, model rehberi ve görselleştirme rehberi.
- `models/`: Standardize edilmiş ONNX model dosyaları (FP32 & INT8).
- `results/`: Ham veriler, CSV raporları ve üretilen akademik figürler.
- `paper/`: Akademik rapor (LaTeX) taslağı ve figür havuzu.
- `npu_gnn_benchmarking.ipynb`: Ana interaktif çalışma arayüzü (Jupyter Notebook).

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Donanım Kıyaslaması (3-Way Comparison)

İşlemcinizdeki **CPU**, **iGPU (Intel Arc)** ve **NPU (AI Boost)** birimlerini aynı model üzerinde kıyaslamak için:

```bash
python analysis/hw_comparison.py --model models/your_model.onnx --iterations 100
```

Bu komut sonucunda `results/hw_comparison/<model_adı>/` altında karşılaştırmalı bir grafik (`hw_comparison_chart.png`) ve tablo üretilir.

## Grafik Optimizasyon Analizi

Bir modelin optimizasyon (Baseline vs Optimized) etkilerini analiz etmek için:

```bash
python analysis/profiling_analyzer.py --baseline results/baseline.json --optimized results/optimized.json
```

## Tam Otomasyon Hattı

Bütün modelleri test edip, analizleri yapıp figürleri rapor dizinine kopyalayan tam akış:

```bash
python run_pipeline.py --iterations 100 --repeats 3
```

## NPU Notu (Intel Core Ultra)

Kod, **Intel OpenVINO** üzerinden NPU'ya erişir. Eğer NPU üzerinde model derleme hatası alırsanız, kütüphane otomatik olarak CPU'ya fallback yapacaktır (BERT modellerinde bazen operatör desteği nedeniyle görülebilir).

---
*Hazırlayan: Antigravity AI Assistant*
