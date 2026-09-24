# ADR 0004 — Transactional outbox for notifications

**Status:** accepted

## Context
The previous JSONL outbox rewrote the whole file to mark deliveries, was unsafe across processes and never actually sent anything. Notifications must exist if and only if the state change committed, and be delivered at most once to each channel.

## Decision
Business code inserts `outbox` rows inside its own transaction with a unique `idempotency_key`. A Celery beat task claims pending rows (`FOR UPDATE SKIP LOCKED` on PostgreSQL), delivers them through adapters (webhook with `Idempotency-Key` header, SMTP/MailHog, console in dev) and marks them `SENT`; failures retry up to five attempts.

## Consequences
+ No lost or phantom notifications; safe with multiple workers.
− Delivery is asynchronous (seconds), acceptable for customer letters.
