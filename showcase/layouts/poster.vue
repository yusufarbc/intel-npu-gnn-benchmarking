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
    <Glossary :terms="terms" />
  </div>
</template>
