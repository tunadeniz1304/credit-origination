"""Browser E2E fixtures: one uvicorn server per session, one browser context per test.

The tests skip cleanly when Playwright or its Chromium build is missing
(``python -m playwright install chromium``).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

from tests.e2e.server import start_server, stop_server

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")

# Responses the UI provokes on purpose; the browser logs them as console errors.
EXPECTED_HTTP_ERRORS: tuple[tuple[str, str], ...] = ()


@pytest.fixture(scope="session")
def e2e_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    proc, base = start_server(tmp_path_factory.mktemp("e2e-server"))
    try:
        yield base
    finally:
        stop_server(proc)


@pytest.fixture(scope="session")
def e2e_browser() -> Iterator[object]:
    manager = playwright_api.sync_playwright().start()
    try:
        try:
            browser = manager.chromium.launch()
        except playwright_api.Error as exc:  # browser binary not downloaded
            pytest.skip(f"Chromium for Playwright is not installed: {str(exc).splitlines()[0]}")
        yield browser
        browser.close()
    finally:
        manager.stop()


@dataclass
class UI:
    """A page plus the console errors it produced."""

    page: object
    base: str
    errors: list[str] = field(default_factory=list)

    def goto_login(self) -> None:
        self.page.goto(self.base + "/")
        self.page.locator("#u").wait_for(state="visible")

    def login(self, username: str, password: str = "Demo123!") -> None:
        self.goto_login()
        self.page.fill("#u", username)
        self.page.fill("#p", password)
        self.page.get_by_role("button", name="Giriş yap").click()
        self.page.get_by_role("button", name="Çıkış").wait_for()

    def logout(self) -> None:
        self.page.get_by_role("button", name="Çıkış").click()
        self.page.locator("#u").wait_for(state="visible")

    def assert_session_cookie_is_http_only(self) -> None:
        cookies = self.page.evaluate("() => document.cookie")
        assert "anil2_session" not in cookies, cookies
        names = {c["name"] for c in self.page.context.cookies()}
        assert "anil2_session" in names  # present, but invisible to scripts


@pytest.fixture
def ui(e2e_browser, e2e_server: str) -> Iterator[UI]:
    context = e2e_browser.new_context(viewport={"width": 1400, "height": 900}, locale="tr-TR")
    context.set_default_timeout(30_000)
    page = context.new_page()
    handle = UI(page=page, base=e2e_server)

    def on_console(message) -> None:
        if message.type != "error":
            return
        url = (message.location or {}).get("url", "")
        if any(status in message.text and part in url for status, part in EXPECTED_HTTP_ERRORS):
            return
        handle.errors.append(f"{message.text} ({url})")

    page.on("console", on_console)
    page.on("pageerror", lambda exc: handle.errors.append(f"pageerror: {exc}"))
    try:
        yield handle
        assert handle.errors == [], "browser console errors:\n" + "\n".join(handle.errors)
    finally:
        context.close()
