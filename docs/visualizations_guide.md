# Visualizations & Academic Evidence Guide

Bu doküman, benchmark pipeline'ı tarafından üretilen grafiklerin akademik bir makalede (IEEE/ACM formatı) nasıl yorumlanması gerektiğini açıklar.

## 1. Roofline Model (`roofline_model.png`)
**Amacı:** Donanımın limitleri ile modelin aritmetik yoğunluğunu (Arithmetic Intensity) ilişkilendirmek.

*   **X Ekseni:** Arithmetic Intensity (FLOPs/Byte).
*   **Y Ekseni:** Achieved Performance (GFLOPS).
*   **Yorum:** GNN modelleri (GCN, GAT) grafiğin solundaki eğimli "Memory-Bound" bölgesinde yer alır. ResNet50 gibi modeller ise sağdaki yatay "Compute-Bound" bölgesine yakındır.
*   **Kanıt:** "GNN'lerin NPU'nun teorik TOPS değerine ulaşamamasının nedeni hesaplama gücü değil, bellek bant genişliği darboğazıdır (Memory Wall)."

## 2. Latency Breakdown (`latency_stacked_100pct.png`)
**Amacı:** Çıkarım süresinin hangi aşamalarda harcandığını yüzdesel olarak göstermek.

*   **Bileşenler:** NPU Compute, CPU Fallback, DMA (Data Transfer), Dispatch/Overhead.
*   **Yorum:** Küçük modellerde ve GNN'lerde "Dispatch" ve "DMA" payının, gerçek hesaplamadan (Compute) çok daha büyük olduğu görülür.
*   **Kanıt:** "Fusion Overhead Paradox" ve "Execution Dispatch Latency" sorunlarının küçük iş yüklerindeki baskınlığı.

## 3. Optimization Efficiency (`fgr_diverging.png`)
**Amacı:** OpenVINO optimizasyonlarının (Fusion, Constant Folding) başarısını ölçmek.

*   **Eşik:** 1.0 (Baseline).
*   **Yorum:** 1.0'dan büyük değerler (Yeşil) başarılı optimizasyonu, 1.0'dan küçük değerler (Kırmızı) optimizasyonun performans kaybına yol açtığını (Paradox) gösterir.
*   **Kanıt:** "Standart derin öğrenme optimizasyon stratejileri, seyrek (sparse) GNN yapılarında her zaman pozitif sonuç vermez."

## 4. Execution Provider Distribution (`provider_fallback_analysis.png`)
**Amacı:** NPU'nun modeldeki operatörlerin ne kadarını desteklediğini göstermek.

*   **Yorum:** "Unknown" veya "CPU" olarak işaretlenen bölgeler, NPU plugin'inin desteklemediği ve CPU'ya geri düşen (Fallback) operatörleri temsil eder.
*   **Kanıt:** "NPU mimarisinin GNN-spesifik operasyonlara (Sparse Gather/Scatter) olan kısıtlı desteği."
