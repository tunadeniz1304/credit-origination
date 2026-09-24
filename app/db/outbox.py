"""Transactional outbox + notification adapters.

Business code calls :func:`enqueue` inside its own transaction, so a
notification exists if and only if the state change committed. The
dispatcher claims pending rows (``FOR UPDATE SKIP LOCKED`` on PostgreSQL),
delivers them through a channel adapter (webhook, SMTP; console in dev) and
marks them ``SENT``. Delivery is idempotent: each message carries a unique
``idempotency_key`` that adapters forward (``Idempotency-Key`` header) and a
message is never re-sent once ``SENT``.
"""

from __future__ import annotations

import smtplib
from collections.abc import Callable
from email.message import EmailMessage
from typing import Any, Protocol

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models import OutboxMessage, utcnow
from app.db.session import session_scope

MAX_ATTEMPTS = 5
logger = get_logger("outbox.dispatcher")


class ChannelAdapter(Protocol):
    name: str

    def send(self, message: OutboxMessage) -> None: ...


class ConsoleAdapter:
    """Development adapter: logs the notification (PII-masked by the log filter)."""

    name = "console"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, message: OutboxMessage) -> None:
        self.sent.append(message.idempotency_key)
        logger.info(
            "notification %s event=%s aggregate=%s channel=%s",
            message.idempotency_key,
            message.event,
            message.aggregate_id,
            message.channel,
        )


class WebhookAdapter:
    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0, client: httpx.Client | None = None) -> None:
        self.url = url
        self._client = client or httpx.Client(timeout=timeout)

    def send(self, message: OutboxMessage) -> None:
        response = self._client.post(
            self.url,
            json={"event": message.event, "aggregate_id": message.aggregate_id, **message.payload},
            headers={"Idempotency-Key": message.idempotency_key},
        )
        response.raise_for_status()


class SmtpAdapter:
    name = "email"

    def __init__(self, host: str, port: int, sender: str) -> None:
        self.host, self.port, self.sender = host, port, sender

    def send(self, message: OutboxMessage) -> None:
        recipient = message.payload.get("to")
        if not recipient:
            raise ValueError("email notification without recipient")
        mail = EmailMessage()
        mail["From"] = self.sender
        mail["To"] = recipient
        mail["Subject"] = message.payload.get("subject", message.event)
        mail["Message-ID"] = f"<{message.idempotency_key}@anil2.local>"
        mail.set_content(message.payload.get("body", ""))
        with smtplib.SMTP(self.host, self.port, timeout=5) as smtp:
            smtp.send_message(mail)


def default_adapters(settings: Settings | None = None) -> dict[str, ChannelAdapter]:
    settings = settings or get_settings()
    console = ConsoleAdapter()
    return {
        "webhook": WebhookAdapter(settings.webhook_url) if settings.webhook_url else console,
        "email": SmtpAdapter(settings.smtp_host, settings.smtp_port, settings.smtp_from)
        if settings.smtp_host
        else console,
        "console": console,
    }


def enqueue(
    session: Session,
    *,
    event: str,
    aggregate_id: str,
    payload: dict[str, Any],
    channel: str = "webhook",
    idempotency_key: str | None = None,
) -> OutboxMessage | None:
    """Insert a message in the caller's transaction; duplicates are ignored."""
    key = idempotency_key or f"{event}:{aggregate_id}"
    existing = session.execute(
        select(OutboxMessage).where(OutboxMessage.idempotency_key == key)
    ).scalar_one_or_none()
    if existing is not None:
        return None
    message = OutboxMessage(
        event=event,
        aggregate_id=aggregate_id,
        payload=payload,
        channel=channel,
        idempotency_key=key,
    )
    try:
        with session.begin_nested():
            session.add(message)
    except IntegrityError:  # concurrent duplicate
        return None
    return message


def dispatch_pending(
    settings: Settings | None = None,
    adapters: dict[str, ChannelAdapter] | None = None,
    limit: int = 100,
    on_sent: Callable[[OutboxMessage], None] | None = None,
) -> dict[str, int]:
    """Deliver pending messages; return counts of sent/failed."""
    settings = settings or get_settings()
    adapters = adapters or default_adapters(settings)
    sent = failed = 0
    with session_scope(settings) as session:
        query = (
            select(OutboxMessage)
            .where(OutboxMessage.status == "PENDING")
            .order_by(OutboxMessage.created_at)
            .limit(limit)
        )
        if session.get_bind().dialect.name == "postgresql":
            query = query.with_for_update(skip_locked=True)
        for message in session.execute(query).scalars():
            adapter = adapters.get(message.channel) or adapters["console"]
            message.attempts += 1
            try:
                adapter.send(message)
            except Exception as exc:
                message.last_error = type(exc).__name__
                if message.attempts >= MAX_ATTEMPTS:
                    message.status = "FAILED"
                failed += 1
                continue
            message.status = "SENT"
            message.delivered_at = utcnow()
            sent += 1
            if on_sent:
                on_sent(message)
    return {"sent": sent, "failed": failed}


def mark_delivered(session: Session, message_id: str) -> bool:
    message = session.get(OutboxMessage, message_id)
    if message is None or message.status == "SENT":
        return False
    message.status = "SENT"
    message.delivered_at = utcnow()
    message.attempts += 1
    return True
