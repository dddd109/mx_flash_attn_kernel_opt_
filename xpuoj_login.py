"""XPUOJ login - saves storage_state to xpuoj_state.json for reuse."""
from playwright.sync_api import sync_playwright

USER = "muxi2026C2032@example.com"
PWD = "MDfm3jQ2eZsrVVAb"

with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=['--no-sandbox'])
    ctx = b.new_context()
    pg = ctx.new_page()
    pg.goto('https://xpuoj.com/login', timeout=60000)
    pg.wait_for_timeout(4000)
    pg.locator('input[type="text"]').first.fill(USER)
    pg.locator('input[type="password"]').fill(PWD)
    pg.locator('form').first.locator('button').first.click()
    pg.wait_for_timeout(6000)
    # check logged in by going to profile
    pg.goto('https://xpuoj.com/u/muxi2026C2032', timeout=40000)
    pg.wait_for_timeout(3000)
    body = pg.inner_text('body')
    if 'muxi2026C2032' in body and 'Login' not in body[:100]:
        ctx.storage_state(path='/root/code/xpuoj_state.json')
        print("LOGIN OK - state saved")
    else:
        print("LOGIN FAILED")
    b.close()
