<template>
  <footer v-if="$nav.currentPage > 1" class="global-footer">
    <div class="footer-text-left">Benchmarking GNN Inference on Intel Core Ultra NPU &mdash; Arabacı et al.</div>
    <div class="slide-page-number">{{ $nav.currentPage }} / {{ $nav.total }}</div>
  </footer>
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
</style>
