from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def load_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Profiling file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]

    if isinstance(payload, dict):
        trace_events = payload.get("traceEvents", [])
        if isinstance(trace_events, list):
            return [event for event in trace_events if isinstance(event, dict)]

    return []


def _extract_duration_us(event: Dict[str, Any]) -> Optional[float]:
    duration = event.get("dur")
    if isinstance(duration, (int, float)) and duration >= 0:
        return float(duration)
    args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
    for key in ("dur", "duration", "duration_us", "op_time_us"):
        value = args.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    return None


def _extract_operator_name(event: Dict[str, Any]) -> Optional[str]:
    args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
    op_name = args.get("op_name") or event.get("op_name")
    if isinstance(op_name, str) and op_name.strip():
        return op_name.strip()

    raw_name = event.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None

    cleaned = raw_name.strip()
    if cleaned.endswith("_kernel_time"):
        cleaned = cleaned.replace("_kernel_time", "")
    if "(" in cleaned:
        cleaned = cleaned.split("(", 1)[0].strip()
    if "::" in cleaned:
        cleaned = cleaned.split("::")[-1].strip()
    return cleaned or None


def _extract_provider(event: Dict[str, Any]) -> str:
    args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
    for key in ("provider", "execution_provider", "exec_provider", "device", "ep"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    name = event.get("name")
    if isinstance(name, str):
        if "CPUExecutionProvider" in name:
            return "CPUExecutionProvider"
        if "OpenVINOExecutionProvider" in name:
            return "OpenVINOExecutionProvider"

    return "Unknown"


def _category_for_operator(op_name: str) -> str:
    name_lower = op_name.lower()
    if any(x in name_lower for x in ("memcpy", "dma", "copy", "transfer")):
        return "dma"
    return "compute"


@dataclass(frozen=True)
class TraceSummary:
    total_ms: float
    compilation_ms: float
    compute_ms: float
    dma_ms: float
    dispatch_ms: float
    provider_time_ms: Dict[str, float]
    cpu_fallback_ms: float
    cpu_fallback_pct: float


def extract_total_inference_time_ms(events: List[Dict[str, Any]], event_name: str = "model_run") -> float:
    def _dur_ms(e: Dict[str, Any]) -> float:
        return float(e.get("dur", 0.0)) / 1000.0

    runs = [_dur_ms(e) for e in events if e.get("name") == event_name]
    if runs:
        return float(np.mean(runs))

    fallback_names = {"SequentialExecutor::Execute", "InferenceSession::Run"}
    runs = [_dur_ms(e) for e in events if e.get("name") in fallback_names]
    if runs:
        return float(np.mean(runs))

    return 0.0


def extract_compilation_time_ms(events: List[Dict[str, Any]], event_name: str = "session_initialization") -> float:
    for event in events:
        if event.get("name") == event_name:
            return float(event.get("dur", 0.0)) / 1000.0
    return 0.0


def iter_operator_events(events: Iterable[Dict[str, Any]]) -> Iterable[Tuple[str, float, str, str]]:
    """Yield (op_name, duration_us, provider, category) for operator-like events."""
    for event in events:
        op_name = _extract_operator_name(event)
        dur_us = _extract_duration_us(event)
        if op_name is None or dur_us is None:
            continue
        provider = _extract_provider(event)
        category = _category_for_operator(op_name)
        yield op_name, dur_us, provider, category


def summarize_trace(
    events: List[Dict[str, Any]],
    *,
    compilation_event_name: str = "session_initialization",
    total_run_event_name: str = "model_run",
) -> TraceSummary:
    total_ms = extract_total_inference_time_ms(events, total_run_event_name)
    compilation_ms = extract_compilation_time_ms(events, compilation_event_name)

    compute_us = 0.0
    dma_us = 0.0
    provider_us: Dict[str, float] = {}

    for _op, dur_us, provider, category in iter_operator_events(events):
        provider_us[provider] = provider_us.get(provider, 0.0) + float(dur_us)
        if category == "dma":
            dma_us += float(dur_us)
        else:
            compute_us += float(dur_us)

    compute_ms = compute_us / 1000.0
    dma_ms = dma_us / 1000.0
    dispatch_ms = max(0.0, total_ms - (compute_ms + dma_ms))

    provider_time_ms = {k: v / 1000.0 for k, v in provider_us.items()}

    cpu_fallback_ms = 0.0
    for provider, ms in provider_time_ms.items():
        if "cpu" in provider.lower():
            cpu_fallback_ms += ms

    cpu_fallback_pct = (cpu_fallback_ms / total_ms * 100.0) if total_ms > 0 else 0.0

    return TraceSummary(
        total_ms=float(total_ms),
        compilation_ms=float(compilation_ms),
        compute_ms=float(compute_ms),
        dma_ms=float(dma_ms),
        dispatch_ms=float(dispatch_ms),
        provider_time_ms=provider_time_ms,
        cpu_fallback_ms=float(cpu_fallback_ms),
        cpu_fallback_pct=float(cpu_fallback_pct),
    )
