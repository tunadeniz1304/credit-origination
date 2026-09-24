# ADR 0005 — Application-level PII encryption with HMAC blind indexes

**Status:** accepted

## Context
TCKN, name, phone, address, e-mail and IBAN are personal data under KVKK. Database-level encryption does not protect against privileged SQL access or backups, yet the platform needs equality lookups (duplicate/velocity checks, fraud rings).

## Decision
Encrypt PII columns with Fernet (`PII_ENCRYPTION_KEY`) in the application layer and store a keyed HMAC-SHA256 blind index of the normalised value (`BLIND_INDEX_KEY`) for equality search. API responses mask identifiers; logs pass a PII masking filter; LLM prompts are pseudonymised. A retention job anonymises rejected/cancelled applicants after `RETENTION_DAYS_REJECTED`.

## Consequences
+ A database dump reveals no clear identifiers; lookups still work.
− No substring search on PII (not needed); key rotation requires re-encryption (documented operational task).
