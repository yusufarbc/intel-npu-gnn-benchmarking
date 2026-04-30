# NPU GNN Benchmarking Methodology

Bu doküman, Intel Core Ultra NPU mimarisi üzerinde Grafik Sinir Ağları (GNN) için uygulanan performans karakterizasyonu metodolojisini açıklar.

## 1. Temel Metrikler

Araştırmamızda, standart gecikme (latency) metriklerinin ötesine geçerek GNN optimizasyon verimliliğini ölçen şu metrikleri kullanıyoruz:

### Fusion Gain Ratio (FGR)
Operatör birleştirme (operator fusion) optimizasyonunun etkinliğini ölçer.
$$FGR = \frac{Latency_{Baseline}}{Latency_{Optimized}}$$
*   **FGR > 1:** Fusion başarılı, performans artışı sağlandı.
*   **FGR < 1:** "Fusion Overhead Paradox" durumu. Fusion maliyeti, hesaplama kazancından fazladır (GNN'lerde sıkça görülür).

### Compilation Efficiency Index (CEI)
Derleme süresinin sağlanan performans artışına oranını temsil eder.
$$CEI = \frac{Compilation\_Time_{Total}}{Latency\_Reduction}$$
Bu metrik, OpenVINO'nun "super-linear compilation time" sorununu GNN modellerinin derinliğiyle ilişkilendirmek için kullanılır.

### Arithmetic Intensity (AI)
Modellerin hesaplama-yoğun (compute-bound) mu yoksa bellek-yoğun (memory-bound) mu olduğunu belirler.
$$AI = \frac{Total\_FLOPs}{Total\_Memory\_Transfer\_Bytes}$$

### NPU Support Ratio
Modeldeki operatörlerin ne kadarının NPU (OpenVINO EP) üzerinde, ne kadarının CPU fallback ile çalıştığını gösterir.
$$Support\_Ratio = \frac{Invocations_{NPU}}{Invocations_{Total}} \times 100$$
Bu metrik, GNN mimarilerinin NPU mimarisiyle uyumluluğunu (mapping efficiency) ölçer.

---

## 2. Enerji Telemetrisi ve Verimlilik
Watt başına performans analizi için Intel RAPL veya HWiNFO logları üzerinden gerçek güç tüketimi (Package Power) verileri kullanılır.
*   **Energy Efficiency (EE):** Giga-Inferences per Joule (GI/J).
*   **Idle vs. Load Power:** NPU'nun aktif olduğu andaki güç sıçraması.

---

## 3. Roofline Performance Model
NPU'nun donanım limitlerini (Teorik GFLOPS ve Bellek Bant Genişliği) modelin gerçek performansı ile kıyaslar. GNN'lerin neden "Memory Wall" (Bellek Duvarı) darboğazına takıldığını görselleştirmek için kullanılır.

## 3. Latency Breakdown Analysis
Toplam çıkarım (inference) süresini şu bileşenlere ayırır:
*   **Compute:** Saf kernel hesaplama süresi.
*   **DMA (Direct Memory Access):** Veri transfer süresi (CPU <-> NPU).
*   **Dispatch/Scheduling:** Kernel fırlatma ve koordinasyon maliyeti.
