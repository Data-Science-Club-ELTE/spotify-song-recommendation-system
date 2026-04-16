"""Capture README screenshots from the running Streamlit app (Playwright)."""
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8501"
OUT = Path(__file__).resolve().parent.parent / "docs" / "images"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(BASE, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(8000)
        page.screenshot(path=str(OUT / "app-home.png"), full_page=True)

        inp = page.get_by_placeholder("e.g., Carve")
        inp.fill("Carve")
        page.get_by_role("button", name="Recommend").click()
        page.wait_for_timeout(12000)
        page.screenshot(path=str(OUT / "app-recommendations.png"), full_page=True)

        browser.close()


if __name__ == "__main__":
    main()
