# npu-graph-opt-benchmarking

Bu proje, **Intel Core Ultra (Meteor Lake)** gibi modern işlemcilerde NPU, GPU ve CPU performansını kıyaslamak ve grafik optimizasyonlarının (operatör füzyonu) etkilerini analiz etmek için tasarlanmıştır.

## Proje Yapısı

Proje, mantıksal bölümlere ayrılarak yeniden organize edilmiştir:

- `engine/`: Benchmark motoru ve donanım seçim mantığı.
- `analysis/`: Veri analizi, grafik üretimi ve donanım kıyaslama araçları.
- `utils/`: Model indirme gibi yardımcı araçlar.
- `models/`: ONNX model dosyaları.
- `results/`: Profiling verileri, CSV raporları ve görsel grafikler.
- `results/archive/`: Eski test sonuçlarının yedekleri.
- `paper/`: Akademik rapor (LaTeX) ve figürler.

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
