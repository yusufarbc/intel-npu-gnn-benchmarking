# NPU GNN Performans Karakterizasyonu Metodolojisi

Bu doküman, Intel Core Ultra NPU mimarisi üzerinde Grafik Sinir Ağları (GNN) için uygulanan bilimsel analiz yöntemlerini detaylandırır.

## 1. Temel Performans Metrikleri

Standart gecikme (latency) ölçümlerinin ötesinde, donanım-yazılım etkileşimini anlamak için şu metrikler kullanılmaktadır:

### Fusion Gain Ratio (FGR)
Derleyici seviyesindeki operatör birleştirme (operator fusion) optimizasyonlarının gerçek hızlanma etkisini ölçer.
$$FGR = \frac{Gecikme_{Baz}}{Gecikme_{Optimize}}$$
*   **FGR < 1:** "Fusion Overhead Paradox" durumu; optimizasyonun ek yükü kazancından fazladır.

### Compilation Efficiency Index (CEI)
Modelin derleme (compilation) süresinin, sağladığı performans artışına oranını temsil eder.
$$CEI = \frac{Derleme\_Suresi}{Gecikme\_Azalimi}$$

### Aritmetik Yoğunluk (AI)
Modellerin hesaplama odaklı (compute-bound) mu yoksa bellek odaklı (memory-bound) mu olduğunu belirler.
$$AI = \frac{Toplam\_FLOP}{Toplam\_Bellek\_Transferi\_Bayt}$$

---

## 2. Deneysel Düzenek ve Veri Seti

*   **Donanım:** Intel Core Ultra 5 125H (Meteor Lake) NPU 3720.
*   **Yazılım:** OpenVINO 2024.1, ONNX Runtime 1.17.
*   **Veri Seti:** **Cora** Alıntı Ağı (2708 düğüm, 5429 kenar). NPU üzerinde statik shape desteği için girişler sabitlenmiş ve padding uygulanmıştır.
*   **Hassasiyet:** FP32 (baseline) ve INT8 (NNCF ile kuantize edilmiş native NPU modu).

---

## 3. Analitik Modeller

### Roofline Performans Modeli
NPU'nun tepe işlem gücü ve bellek bant genişliği limitlerini modelin gerçek performansıyla kıyaslar. GNN'lerin neden "Bellek Duvarı" (Memory Wall) darboğazına takıldığını kanıtlar.

### Latency Breakdown (Gecikme Ayrıştırması)
Toplam çıkarım süresini şu üç ana bileşene ayırır:
1.  **Compute:** Saf kernel hesaplama süresi.
2.  **DMA:** CPU ve NPU arasındaki veri transfer süresi.
3.  **Dispatch:** Kernel fırlatma ve zamanlama maliyeti.

---
*Son Güncelleme: 2 Mayıs 2026*
