# IEEE HPEC 2026 Camera-Ready Checklist

Paper 101: **Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis**

Deadline: **September 4, 2026**

This checklist separates repository-verified items from account-dependent submission steps. Do not edit the PDF after it passes IEEE PDF eXpress; if the manuscript changes, run PDF eXpress again.

## Repository-verified preparation

- [x] Uses the IEEE `conference` template (`IEEEtran`).
- [x] Main technical content ends on page 6; acknowledgments and references follow and are excluded from the HPEC six-page limit.
- [x] Paper is US Letter, two-column, unencrypted PDF 1.5.
- [x] All fonts are embedded and subset Type 1 fonts.
- [x] PDF contains no JavaScript, forms, attachments, bookmarks, or link annotations.
- [x] Title, author order, affiliations, abstract, software versions, figures, and headline numbers are synchronized with the repository and poster.
- [x] GenAI disclosure identifies the systems, affected sections, type of assistance, subsequent human revision, and states that no AI system generated data, results, or figures.
- [x] Reviewer terminology and clarity requests are addressed: NPU, SpMM/SpDMM, subprocess loading, latency, iGPU, failed outcomes, and the Figure 3 `Other` category.
- [x] The former CPU-fallback heatmap is replaced in the paper and poster by a device-assignment exception table.
- [x] The six-slide virtual poster covers the challenge, approach, experimental setup, results, guidance, and limitations.
- [x] A local offline poster backup is generated as `showcase/poster-export.pdf` by `npm run export`.

## Exact CMT metadata

Enter this information in CMT exactly as it should appear in the proceedings.

### Title

Benchmarking GNN Inference on the Intel Core Ultra NPU: A Latency, Quantization, and Energy Analysis

### Author order

1. Yusuf Talha ARABACI - Department of Software Engineering, Karabuk University, Karabuk, Turkey - yusuftalhaarabaci@hotmail.com
2. Emrullah DEMİRAL - Department of Software Engineering, Karabük University, Karabük, Turkey - emrullahdemiral@karabuk.edu.tr
3. Ömer Faruk ACAR - Department of Software Engineering, Karabük University, Karabük, Turkey - farukacar@karabuk.edu.tr

Use the Turkish characters shown in the PDF/CMT interface: **Emrullah DEMİRAL** and **Ömer Faruk ACAR**.

### Abstract

Neural Processing Units (NPUs) in client System-on-Chips (SoCs) target low-power inference, but their performance on sparse Graph Neural Networks (GNNs) remains underexplored. We benchmark 14 GNN, CNN, and transformer models across the CPU, integrated GPU (iGPU), and NPU of an Intel Core Ultra platform using OpenVINO and three Open Graph Benchmark datasets. The NPU accelerates dense vision models (MobileNetV2: 1.90 ms, 4.5-11.4x speedup over CPU) but yields negligible gains for memory-bound GNNs. NPU INT8 quantization causes compilation failures in three architectures and a 2.2x latency regression for SGC. CPU and iGPU backends operate within a 9-12.5 W power range, with INT8 providing at most an 18% energy reduction. The observed performance gap is consistent with a mismatch between the NPU's streaming-dataflow design and the irregular memory accesses of GNNs. We recommend the iGPU for GNNs and the NPU for FP32 vision models. The suite is available at https://yusufarbc.github.io/intel-npu-gnn-benchmarking/.

## Required external actions

These steps require the corresponding author's conference accounts or conference ID and cannot be completed from the repository.

- [ ] Proofread the final PDF one last time with all authors.
- [ ] Confirm that every author has reviewed and can defend every statement, table, figure, result, interpretation, and conclusion.
- [ ] Complete a final plagiarism and attribution review: all reused words, methods, data, figures, and results are explicitly cited or licensed as required.
- [ ] Upload `paper/paper.pdf` to IEEE PDF eXpress using the HPEC conference ID received by email.
- [ ] Confirm that the PDF eXpress report explicitly says the file passed.
- [ ] Download/use the passed PDF without modifying it afterward.
- [ ] In the HPEC 2026 CMT author role, create the camera-ready submission for Paper 101.
- [ ] Paste the title, author order, affiliations, and abstract from the exact metadata above.
- [ ] Upload the PDF eXpress-approved PDF to CMT by September 4, 2026.
- [ ] Submit the IEEE electronic Copyright Form (eCF) through CMT.
- [ ] Confirm the first-author student-paper flag if applicable.
- [ ] Complete a full presenting-author registration and retain the confirmation.
- [ ] Confirm the camera-ready PDF, copyright form, and metadata are visible in CMT before the deadline.

## Poster-session readiness

- [ ] Open the live poster in a private/incognito browser: https://yusufarbc.github.io/intel-npu-gnn-benchmarking/
- [ ] Confirm all six slides and every `Term Notes` control work.
- [ ] Confirm the QR code opens the repository.
- [ ] Keep `showcase/poster-export.pdf` open as an offline backup.
- [ ] Test Zoom screen sharing at 100% browser zoom and verify both charts are readable.
- [ ] Prepare a short explanation of the challenge, approach, setup, results, and limitations for breakout-room questions.
