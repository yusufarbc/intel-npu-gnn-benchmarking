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

## 4. Teknik Kısıtlamalar ve Karşılaşılan Hatalar

Benchmark sürecinde bazı modellerde mimari uyuşmazlıklar tespit edilmiştir. Bu durumlar akademik raporlamada "NPU Donanım/Yazılım Olgunluk Analizi" kapsamında sunulmalıdır:

*   **GAT (Graph Attention Network):** OpenVINO Execution Provider (EP), NPU üzerinde GAT modellerini derlerken `Output names mismatch between OpenVINO and ONNX` hatası vermektedir. Bu durum, NPU plugin'inin dikkat mekanizmasındaki (attention) alt grafiklerin çıktı isimlendirmelerini henüz tam olarak eşleştiremediğini göstermektedir. Bu model, NPU kısıtlaması nedeniyle CPU fallback modunda çalıştırılmaktadır.
*   **GraphTransformer:** Modelin `Self-Attention` katmanlarındaki dinamik tensör şekilleri ve `Sub` operatörü üzerindeki boyut uyuşmazlıkları (Incompatible dimensions), statik kuantizasyon (PTQ) sürecinde "Shape Inference" hatalarına yol açmıştır. (Giriş şekilleri optimize edilerek bazı versiyonlarda aşılmıştır).
*   **BERT-tiny:** Opset yükseltme (Opset 11 -> 13) sırasında `Unsqueeze` operatörünün parametre sınırları dışında kalması nedeniyle kuantizasyon motoru (NNCF/ONNX) tarafından reddedilmiştir.
*   **BERT-tiny (GPU):** OpenVINO GPU backend, `bert-tiny_fp32` için derleme sırasında hata vermektedir. Bu yüzden `hw_comparison` aşamasında GPU koşumu otomatik olarak atlanır; CPU ve NPU sonuçları raporlanır.

**Akademik Not:** GAT modelinde karşılaşılan derleme hatası ve diğer modellerdeki kuantizasyon zorlukları, Intel Core Ultra NPU mimarisinin (NPU 3720 IP) özellikle dinamik attention mekanizmalarına sahip grafik yapıları için yazılım kütüphanesi seviyesinde (OpenVINO NPU Plugin) geliştirilmeye muhtaç alanlarını kanıtlayan kritik "araştırma bulguları" (research findings) olarak nitelendirilmelidir.

---
*Son Güncelleme: 30 Nisan 2026*

### Intel Core Ultra NPU (VPUX37XX) Donanım ve Derleyici Limitasyonları

Benchmarking çalışmaları sırasında Intel NPU çekirdek sürücüsü (OpenVINO NPU Plugin) seviyesinde bazı mimari kısıtlamalar tespit edilmiştir:

1.  **Negatif Post-Shift Kuantizasyon Hatası:**
    Özellikle Message-Passing tabanlı bazı GNN modellerinin (örn. `APPNP_int8.onnx`) INT8 kuantizasyon sürecinde aşırı uçlarda ölçekleme faktörleri (scale factors) üretilmektedir. NPU donanımı (VPUX37XX mimarisi) derleme esnasında şu hatayı fırlatmaktadır:
    `ConvertIEToVPUNCE Pass failed : Encountered an attempt to approximate 56674.56 as mult = 28337, shift = 0, postShift = -1 ... but postShift is not supported`
    Bu hata, donanımın o spesifik tensor için hesaplanan kuantizasyon `post-shift` değerini (örn: `-1`) donanımsal olarak desteklememesinden kaynaklanır. Bu modellerin çalışması OpenVINO tarafından otomatik olarak iptal edilir.
    
2.  **Intel Graphics Compiler (IGC) Çökmeleri:**
    NPU ve GPU backend'leri tarafından ortak olarak kullanılan IGC derleyicisi, bazı INT8 Quantized GNN grafikleri işlenirken `intersects with V37` şeklinde ölümcül bellek segmentasyon hataları (Memory Access / Intersection Error) vermektedir. Pipeline bu hataların tüm Python sürecini çökertmemesi için koruma mekanizmaları (GPU Bypass) ile donatılmıştır.

Bu kısıtlamalar bir yazılım hatasından (bug) ziyade, GNN'lerin standart CNN ve Transformer modelleri için tasarlanmış NPU donanım/yazılım yığınlarıyla mevcut uyumsuzluklarını gösteren kıymetli araştırma bulgularıdır.