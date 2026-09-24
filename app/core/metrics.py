"""Prometheus metrics registry for the platform.

All collectors live on a dedicated registry so tests can import this module
repeatedly without duplicate-registration errors. ``/metrics`` exposes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

if TYPE_CHECKING:
    from app.agents.llm_service import LLMCallRecord

REGISTRY = CollectorRegistry(auto_describe=True)

APPLICATIONS_SUBMITTED = Counter(
    "anil2_applications_submitted_total", "Submitted applications", registry=REGISTRY
)
STATE_TRANSITIONS = Counter(
    "anil2_state_transitions_total",
    "Application state transitions",
    ["to_state"],
    registry=REGISTRY,
)
DECISIONS = Counter(
    "anil2_decisions_total", "Decision engine outcomes", ["outcome"], registry=REGISTRY
)
STAGE_SECONDS = Histogram(
    "anil2_stage_duration_seconds",
    "Pipeline stage durations",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)
CIRCUIT_STATE = Gauge(
    "anil2_circuit_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["service"],
    registry=REGISTRY,
)
QUEUE_DEPTH = Gauge("anil2_review_queue_depth", "Open specialist review items", registry=REGISTRY)
LLM_CALLS = Counter("anil2_llm_calls_total", "LLM calls", ["task", "mode"], registry=REGISTRY)
LLM_USAGE = Counter("anil2_llm_tokens_total", "LLM tokens", ["kind"], registry=REGISTRY)
LLM_FAILURES = Counter(
    "anil2_llm_failures_total", "LLM failures by kind", ["kind"], registry=REGISTRY
)
LLM_LATENCY = Histogram(
    "anil2_llm_latency_seconds",
    "LLM call latency",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),
    registry=REGISTRY,
)
HTTP_REQUESTS = Counter(
    "anil2_http_requests_total", "HTTP requests", ["method", "status"], registry=REGISTRY
)


def observe_llm_call(rec: LLMCallRecord) -> None:
    LLM_CALLS.labels(task=rec.task, mode=rec.mode).inc()
    if rec.error_kind:
        LLM_FAILURES.labels(kind=rec.error_kind).inc()
    if rec.prompt_tokens:
        LLM_USAGE.labels(kind="prompt").inc(rec.prompt_tokens)
    if rec.completion_tokens:
        LLM_USAGE.labels(kind="completion").inc(rec.completion_tokens)
    if rec.mode != "demo":
        LLM_LATENCY.observe(rec.latency_ms / 1000.0)


def render_latest() -> bytes:
    return generate_latest(REGISTRY)
