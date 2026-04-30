# NPU GNN Benchmarking Model Guide

Bu doküman, Intel Core Ultra NPU benchmark çalışmasında kullanılan Grafik Sinir Ağları (GNN) ve karşılaştırma temelli (baseline) modellerin detaylarını içerir.

## 1. Grafik Sinir Ağları (GNNs)

Bu modeller, NPU'nun seyrek (sparse) veri yapıları ve düzensiz bellek erişim desenleri üzerindeki performansını ölçmek için seçilmiştir.

| Model | Mimari Özelliği | Seçilme Nedeni |
| :--- | :--- | :--- |
| **GCN** | Spectral Convolution | GNN literatürünün temel baseline'ı. |
| **GAT** | Attention Mechanism | Dinamik ağırlıklandırmanın NPU üzerindeki yükünü ölçer. |
| **GraphSAGE** | Inductive Learning | Büyük graflar için örnekleme (sampling) verimliliğini test eder. |
| **GIN** | Isomorphism Network | Maksimum ekspresivite ve karmaşık agregasyon testi. |
| **SGC** | Simplified Convolution | Gereksiz lineer olmayan katmanların kaldırılmasının etkisini ölçer. |
| **APPNP** | Personalized PageRank | Uzak komşuluk ilişkilerinin (multi-hop) NPU bellek yönetimine etkisi. |
| **GraphTransformer**| Self-Attention | GNN ve Transformer hibrit yapılarının NPU uyumluluğu. |

## 2. Karşılaştırma Modelleri (Baselines)

NPU'nun asıl güçlü olduğu alanlar (Dense CNN) ile GNN'ler arasındaki farkı göstermek için kullanılır.

*   **ResNet50 (FP32/INT8):** Standart evrişimli sinir ağı (CNN). NPU'nun "compute-bound" senaryolardaki tavan performansını (Peak TOPS) gösterir.
*   **MobileNetV2 (FP32/INT8):** Edge cihazlar için optimize edilmiş hafif CNN. Verimlilik kıyası için kullanılır.
*   **BERT-tiny (FP32/INT8):** Transformer mimarisinin en küçük hali. "Fusion Overhead Paradox"u NLP bağlamında kanıtlar.

## 3. Hassasiyet ve Versiyonlar

Her model iki farklı hassasiyet (precision) seviyesinde test edilmektedir:

1.  **FP32 (Floating Point 32):** Orijinal hassasiyet. NPU'nun en az verimli olduğu ama doğruluğun en yüksek olduğu mod.
2.  **INT8 (Integer 8):** NNCF (Neural Network Compression Framework) ile kuantize edilmiş versiyon. NPU'nun donanım hızlandırıcılarını (Movidius VPU/NPU IP) tam kapasite kullandığı "native" mod.

## 4. Teknik Kısıtlamalar ve Kuantizasyon Analizi

Benchmark sürecinde bazı modellerin (GraphTransformer ve BERT) INT8 versiyonları üretilememiştir. Bu durum akademik raporlamada şu teknik gerekçelerle sunulmalıdır:

*   **GraphTransformer:** Modelin `Self-Attention` katmanlarındaki dinamik tensör şekilleri ve `Sub` operatörü üzerindeki boyut uyuşmazlıkları (Incompatible dimensions), statik kuantizasyon (PTQ) sürecinde "Shape Inference" hatalarına yol açmıştır.
*   **BERT-tiny:** Opset yükseltme (Opset 11 -> 13) sırasında `Unsqueeze` operatörünün parametre sınırları dışında kalması nedeniyle kuantizasyon motoru (NNCF/ONNX) tarafından reddedilmiştir.

**Akademik Not:** Bu modellerin sadece FP32 olarak sunulması, NPU donanım kütüphanelerinin (OpenVINO/NPU Plugin) henüz çok karmaşık ve dinamik grafik yapıları için tam olgunluğa erişmediğini kanıtlayan bir "bulgu" (finding) olarak makalede kullanılabilir.

---
*Son Güncelleme: 30 Nisan 2026*
