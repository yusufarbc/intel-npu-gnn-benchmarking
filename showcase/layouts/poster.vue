<script setup>
defineProps({
  terms: {
    type: Array,
    default: () => [],
  },
})
</script>

<template>
  <div class="slidev-layout default">
    <slot />
    <Glossary v-if="$nav.currentPage > 1 && terms && terms.length > 0" :terms="terms" />
  </div>
</template>
