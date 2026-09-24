"""Suite-wide isolation: no dotenv, no network, demo LLM, temp artifact dirs.

Environment variables are set at import time (before any ``app`` import) so
the cached settings, the database engine and the dispatcher all see the test
configuration.
"""

from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="anil2-tests-")
os.environ.update(
    {
        "ANIL2_NO_DOTENV": "1",
        "APP_ENV": "test",
        "LLM_MODE": "demo",
        "LLM_API_KEY": "",
        "TASK_QUEUE_BACKEND": "inline",
        "CIRCUIT_STATE_BACKEND": "memory",
        "RATE_LIMIT_ENABLED": "false",
        # TestClient talks plain http://testserver: Secure cookies would never be sent back.
        "COOKIE_SECURE": "false",
        "DATABASE_URL": f"sqlite:///{os.path.join(_TMP, 'test.db')}",
        "REPORT_OUTPUT_DIR": os.path.join(_TMP, "reports"),
        "RESULT_STORE_DIR": os.path.join(_TMP, "results"),
        "UPLOAD_DIR": os.path.join(_TMP, "uploads"),
    }
)
