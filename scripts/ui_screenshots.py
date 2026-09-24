"""Browser E2E check of the web UI with Playwright; saves README screenshots.

Logs in as the applicant, the specialist and the model manager, opens the
main views and fails on console errors. Screenshots go to ``docs/img/``.

Usage: pip install playwright && playwright install chromium
       python scripts/ui_screenshots.py [--base http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(args.base + "/")
        page.wait_for_selector("text=Demo kullanıcılar")
        page.screenshot(path=str(OUT / "login.png"))

        def login(label: str) -> None:
            context.clear_cookies()  # the session lives in an HttpOnly cookie
            page.goto(args.base + "/")
            page.get_by_role("button", name=label, exact=True).click()
            page.wait_for_timeout(1500)

        login("Başvuran")
        page.wait_for_selector("text=Başvurularım")
        rows = page.locator("tbody tr.clickable:visible")
        if rows.count():
            rows.first.click()
            page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "applicant_portal.png"))

        login("Uzman")
        page.wait_for_selector("text=İnceleme kuyruğu")
        queue = page.locator("button.queue-open:visible")
        if queue.count():
            queue.first.click()
            page.wait_for_timeout(1500)
            page.get_by_role("tab", name="Karar + SHAP").click()
            page.wait_for_timeout(1200)
            page.locator("text=SHAP katkıları").first.scroll_into_view_if_needed()
            page.mouse.wheel(0, -250)
            page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "workbench.png"))

        login("Model yöneticisi")
        page.wait_for_selector("text=Toplam başvuru")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "dashboard.png"))
        page.get_by_role("tab", name="Model doğrulama").click()
        page.wait_for_selector("#val-set")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "model_validation.png"), full_page=True)
        browser.close()
    real = [e for e in errors if "favicon" not in e and "404" not in e]
    for e in real:
        print("console error:", e)
    print("screenshots:", sorted(x.name for x in OUT.glob("*.png")))
    return 1 if real else 0


if __name__ == "__main__":
    sys.exit(main())
