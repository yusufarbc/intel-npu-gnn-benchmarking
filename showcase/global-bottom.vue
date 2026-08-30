<template>
  <footer v-if="$nav.currentPage > 1" class="global-footer">
    <div class="footer-text-left">Benchmarking GNN Inference on Intel Core Ultra NPU &mdash; Arabacı et al.</div>
    <div class="slide-page-number">{{ $nav.currentPage }} / {{ $nav.total }}</div>
  </footer>
  <!-- Slidev's PDF renderer can omit interactive fixed controls on alternating
       pages. This non-interactive twin remains behind the live Glossary button
       so every exported/offline slide still identifies the available notes. -->
  <div v-if="$nav.currentPage > 1" class="term-notes-export-fallback" aria-hidden="true">
    <span>&#x1F4A1;</span><span>Term Notes</span>
  </div>
</template>

<style scoped>
.global-footer {
  position: absolute;
  bottom: 0.8rem;
  left: 2rem;
  right: 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  pointer-events: none;
  border-top: 1px solid var(--color-slate-200);
  padding-top: 0.4rem;
  z-index: 50;
}

.footer-text-left {
  font-size: 0.65rem;
  color: var(--color-slate-500);
  font-family: 'Inter', sans-serif;
  max-width: 65%;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.slide-page-number {
  font-size: 0.65rem;
  color: var(--color-slate-500);
  font-family: 'Inter', sans-serif;
  font-weight: 600;
}

.term-notes-export-fallback {
  position: absolute;
  bottom: 0.65rem;
  right: 5.5rem;
  z-index: 49;
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.15rem 0.5rem;
  border: 1px solid var(--color-slate-300);
  border-radius: 4px;
  background: var(--color-slate-100);
  color: var(--color-blue);
  font-family: 'Inter', sans-serif;
  font-size: 0.65rem;
  font-weight: 600;
  line-height: 1.4;
  pointer-events: none;
}
</style>
