"""XPUOJ submit via xvfb + xclip system clipboard (reliable for Monaco large code).
Run under xvfb-run:  xvfb-run -a python3 xpuoj_submit_clip.py <kernel.cu>
"""
from playwright.sync_api import sync_playwright
import sys, time, subprocess, os

kernel = sys.argv[1]
with open(kernel) as f:
    code = f.read()

# set system clipboard via xclip
p = subprocess.run(['xclip', '-selection', 'clipboard'], input=code.encode(), timeout=10)
print("xclip set, rc=", p.returncode)

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=False, args=['--no-sandbox'])  # headful on xvfb
    ctx = b.new_context(storage_state='/root/code/xpuoj_state.json')
    pg = ctx.new_page()
    pg.goto('https://xpuoj.com/contest/11/problem/1', timeout=60000)
    pg.wait_for_timeout(8000)

    editor = pg.locator('.monaco-editor').first
    editor.click()
    pg.wait_for_timeout(500)
    pg.keyboard.press('Control+A')
    pg.wait_for_timeout(300)
    pg.keyboard.press('Control+V')
    pg.wait_for_timeout(3000)

    ln = pg.evaluate("()=>{const t=document.querySelector('textarea');return t?t.value.length:0}")
    print("code length:", ln, "/", len(code))
    if ln < len(code) * 0.9:
        print("WARNING: code may be incomplete")
        pg.screenshot(path='/tmp/xpuoj_fill_fail.png')

    # find the real Submit button (workspace footer / dialog). Click the LAST one with text Submit
    btns = pg.locator('button:has-text("Submit")')
    n = btns.count()
    print("submit buttons:", n)
    # click last (footer submit)
    btns.last.click()
    pg.wait_for_timeout(12000)
    print("after submit URL:", pg.url)
    body = pg.inner_text('body')
    import re
    m = re.search(r'/s/(\d+)', pg.url)
    sid = m.group(1) if m else None
    if not sid:
        m2 = re.search(r'[Ss]ubmission[ #]*#?(\d+)', body)
        sid = m2.group(1) if m2 else None
    print("SUBMISSION_ID:", sid or "unknown")
    pg.screenshot(path='/tmp/xpuoj_result.png')
    b.close()
