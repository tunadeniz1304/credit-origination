"""Credit application state machine.

``TASLAK → GONDERILDI → BELGE_BEKLENIYOR ↔ BELGE_INCELEMEDE → VERI_TOPLANIYOR →
KARAR_MOTORU → (OTOMATIK_ONAY | OTOMATIK_RET | UZMAN_INCELEMESI) →
TEKLIF_SUNULDU → TEKLIF_KABUL → SOZLESME_HAZIR → KULLANDIRILDI`` plus
``IPTAL``, ``REDDEDILDI`` (human rejection) and ``ITIRAZ_INCELEMESI``
(KVKK art. 11 objection). Any transition not listed is rejected.
"""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    TASLAK = "TASLAK"
    GONDERILDI = "GONDERILDI"
    BELGE_BEKLENIYOR = "BELGE_BEKLENIYOR"
    BELGE_INCELEMEDE = "BELGE_INCELEMEDE"
    VERI_TOPLANIYOR = "VERI_TOPLANIYOR"
    KARAR_MOTORU = "KARAR_MOTORU"
    OTOMATIK_ONAY = "OTOMATIK_ONAY"
    OTOMATIK_RET = "OTOMATIK_RET"
    UZMAN_INCELEMESI = "UZMAN_INCELEMESI"
    REDDEDILDI = "REDDEDILDI"
    ITIRAZ_INCELEMESI = "ITIRAZ_INCELEMESI"
    TEKLIF_SUNULDU = "TEKLIF_SUNULDU"
    TEKLIF_KABUL = "TEKLIF_KABUL"
    SOZLESME_HAZIR = "SOZLESME_HAZIR"
    KULLANDIRILDI = "KULLANDIRILDI"
    IPTAL = "IPTAL"


S = State
TRANSITIONS: dict[State, frozenset[State]] = {
    S.TASLAK: frozenset({S.GONDERILDI, S.IPTAL}),
    S.GONDERILDI: frozenset({S.BELGE_BEKLENIYOR, S.BELGE_INCELEMEDE, S.IPTAL}),
    S.BELGE_BEKLENIYOR: frozenset({S.BELGE_INCELEMEDE, S.IPTAL}),
    S.BELGE_INCELEMEDE: frozenset({S.BELGE_BEKLENIYOR, S.VERI_TOPLANIYOR, S.IPTAL}),
    S.VERI_TOPLANIYOR: frozenset({S.KARAR_MOTORU, S.IPTAL}),
    S.KARAR_MOTORU: frozenset({S.OTOMATIK_ONAY, S.OTOMATIK_RET, S.UZMAN_INCELEMESI}),
    S.OTOMATIK_ONAY: frozenset({S.TEKLIF_SUNULDU}),
    S.OTOMATIK_RET: frozenset({S.ITIRAZ_INCELEMESI}),
    S.UZMAN_INCELEMESI: frozenset({S.TEKLIF_SUNULDU, S.REDDEDILDI, S.BELGE_BEKLENIYOR, S.IPTAL}),
    S.REDDEDILDI: frozenset({S.ITIRAZ_INCELEMESI}),
    S.ITIRAZ_INCELEMESI: frozenset({S.TEKLIF_SUNULDU, S.REDDEDILDI, S.UZMAN_INCELEMESI}),
    S.TEKLIF_SUNULDU: frozenset({S.TEKLIF_KABUL, S.IPTAL}),
    S.TEKLIF_KABUL: frozenset({S.SOZLESME_HAZIR}),
    S.SOZLESME_HAZIR: frozenset({S.KULLANDIRILDI, S.IPTAL}),
    S.KULLANDIRILDI: frozenset(),
    S.IPTAL: frozenset(),
}

TERMINAL: frozenset[State] = frozenset(s for s, targets in TRANSITIONS.items() if not targets)
ADVERSE: frozenset[State] = frozenset({S.OTOMATIK_RET, S.REDDEDILDI})
IN_PROGRESS: frozenset[State] = frozenset(
    {S.GONDERILDI, S.BELGE_INCELEMEDE, S.VERI_TOPLANIYOR, S.KARAR_MOTORU}
)

STATE_LABELS: dict[State, str] = {
    S.TASLAK: "Taslak",
    S.GONDERILDI: "Gönderildi",
    S.BELGE_BEKLENIYOR: "Belge Bekleniyor",
    S.BELGE_INCELEMEDE: "Belge İncelemede",
    S.VERI_TOPLANIYOR: "Veri Toplanıyor",
    S.KARAR_MOTORU: "Karar Motoru",
    S.OTOMATIK_ONAY: "Otomatik Onay",
    S.OTOMATIK_RET: "Otomatik Ret",
    S.UZMAN_INCELEMESI: "Uzman İncelemesi",
    S.REDDEDILDI: "Reddedildi",
    S.ITIRAZ_INCELEMESI: "İtiraz İncelemesi",
    S.TEKLIF_SUNULDU: "Teklif Sunuldu",
    S.TEKLIF_KABUL: "Teklif Kabul Edildi",
    S.SOZLESME_HAZIR: "Sözleşme Hazır",
    S.KULLANDIRILDI: "Kullandırıldı",
    S.IPTAL: "İptal",
}


class InvalidTransitionError(ValueError):
    """Raised for a transition that the state machine does not allow."""

    def __init__(self, current: State, target: State) -> None:
        super().__init__(f"invalid transition {current.value} -> {target.value}")
        self.current = current
        self.target = target


def can_transition(current: State | str, target: State | str) -> bool:
    return State(target) in TRANSITIONS[State(current)]


def assert_transition(current: State | str, target: State | str) -> None:
    if not can_transition(current, target):
        raise InvalidTransitionError(State(current), State(target))
