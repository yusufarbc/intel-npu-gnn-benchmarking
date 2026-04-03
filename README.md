# npu-graph-opt-benchmarking

Bu proje, "Ileri Algoritma Analizi" kapsaminda su soruyu ampirik olarak incelemek icin tasarlanmistir:

> NPU mimarilerinde DAG tabanli operator fusion (node merging), asimptotik hesaplama karmasikligini degistirmese de (
> compute tarafinda yaklasik O(N + E)), I/O baskisini ve memory wall etkisini calisma zamaninda nasil degistirir?

## Proje Yapisi

- `models/`: ONNX model dosyalari
- `scripts/`: benchmark kodlari
- `results/`: profiling JSON, CSV ozetleri ve grafik ciktilari

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Calistirma

```bash
python scripts/benchmark_npu.py --model models/your_model.onnx --iterations 100
```

## Script Ne Yapar?

`scripts/benchmark_npu.py` iki ayri modda test yapar:

- Baseline: `GraphOptimizationLevel.ORT_DISABLE_ALL`
- Optimized: `GraphOptimizationLevel.ORT_ENABLE_EXTENDED`

Her mod icin:

- 5 warmup + 100 olcum iterasyonu calistirir
- Ortalama gecikme (ms) ve standart sapma hesaplar
- ONNX Runtime profiling verisini JSON olarak kaydeder
- Sonuclari CSV ve bar chart olarak disari verir

## Ciktilar

`results/` altinda olusan dosyalar:

- `baseline_profiling.json`
- `optimized_profiling.json`
- `performance_summary.csv`
- `performance_comparison.png`

## Operator Bazli Profiling Analizi

Toplam latency'e ek olarak operator seviyesinde sure kirilimi almak icin:

```bash
python scripts/parse_operator_breakdown.py
```

Hoca beklentisine uygun alias script:

```bash
python scripts/parse_profiling.py
```

Bu script, varsayilan olarak `results/baseline_profiling.json` ve `results/optimized_profiling.json` dosyalarini okur ve su ciktilari uretir:

- `operator_breakdown_by_mode.csv` (mode + operator bazli toplam/ortalama sure)
- `operator_breakdown.csv` (baseline vs optimized karsilastirma)
- `operator_breakdown_topk.png` (en agir operatorler icin karsilastirma grafigi)
- `operator_top5_speedup.csv` (en cok hizlanan ilk 5 operator)
- `disappeared_operators.csv` (fuzyon sonrasi kaybolan/birlesen operatorler)
- `operator_count_delta.csv` (operator invocation sayisi degisimi)

## Olceklenebilirlik ve Roofline Analizi

Farkli model boyutlariyla tekrarli calisma ve roofline ozet matrisi icin:

```bash
python scripts/run_scalability_study.py --repeats 3 --iterations 100
```

Bu script:

- `models/` altindaki tum `.onnx` dosyalarini tarar
- her model icin baseline/optimized benchmark'i `repeats` kadar tekrarlar
- ortalama, std, %95 CI, speedup ve `c` faktor orani hesaplar
- dugum sayisi degisimi (`|V|`) raporlar
- basit roofline metrikleri (AI, ridge point, memory/compute-bound sinifi) uretir

Uretilen ozet ciktilar:

- `results/scalability_matrix.csv`
- `results/scalability_speedup.png`
- `results/scalability_latency.png`

Donaniminiza gore roofline parametrelerini elle verebilirsiniz:

```bash
python scripts/run_scalability_study.py --peak-compute-gflops 1600 --peak-bandwidth-gbps 34
```

## Tek Komut Otomasyon (Reproducibility)

Tum akisi tek komutta calistirmak icin:

```bash
python scripts/run_all.py --repeats 3 --iterations 100
```

Bu komut sirasiyla:

1. ilk modeli kullanarak benchmark calistirir
2. profiling parse eder
3. tum modellerde olceklenebilirlik calismasi yapar
4. `results/` altindaki PNG dosyalarini `paper/figures/` altina kopyalar

Opsiyonel roofline parametreleri:

```bash
python scripts/run_all.py --peak-compute-gflops 1600 --peak-bandwidth-gbps 34
```

## Paper Yapisi

`paper/` klasoru akademik raporlama icin olusturuldu:

- `paper/main.tex`
- `paper/references.bib`
- `paper/figures/`

Grafikleri rapora eklemek icin `results/` altindaki PNG dosyalarini `paper/figures/` altina kopyalayabilirsiniz.

## NPU Provider Fallback

Kod, su sirayi hedefleyerek execution provider secer:

1. `OpenVINOExecutionProvider`
2. `QNNExecutionProvider`
3. `DmlExecutionProvider`
4. `CUDAExecutionProvider`
5. `CPUExecutionProvider`

Boylece yerel NPU/ivmelendirici varsa kullanilir, yoksa otomatik fallback ile benchmark yine calisir.
