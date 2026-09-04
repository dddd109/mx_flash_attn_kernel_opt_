"""XPUOJ auto-submit via Monaco model API.
Usage: python3 xpuoj_submit.py <kernel.cu>
Requires xpuoj_state.json (login). Prints submission id + polls result briefly.
"""
from playwright.sync_api import sync_playwright
import sys, time, re

kernel = sys.argv[1]
with open(kernel) as f:
    code = f.read()

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, args=['--no-sandbox'])
    ctx = b.new_context(storage_state='/root/code/xpuoj_state.json')
    pg = ctx.new_page()
    pg.goto('https://xpuoj.com/contest/11/problem/1', timeout=60000)
    pg.wait_for_timeout(8000)

    # set all monaco models to our code
    r = pg.evaluate("""(nc) => {
        const out = [];
        if (window.monaco && window.monaco.editor) {
            for (const m of window.monaco.editor.getModels()) { m.setValue(nc); out.push(m.getValueLength()); }
        }
        return out;
    }""", code)
    print("models set:", r)
    pg.wait_for_timeout(800)

    # ensure language CUDA Maca - it was active. The page shows 'CUDA Maca' in workspace header.
    # Click the real submit: there's a top-bar Submit and workspace. Use the main submit button.
    # Try the one NOT 'Submit settings'. Click last 'Submit' exact-text button.
    submit_btn = pg.locator('button:text-is("Submit")').last
    try:
        submit_btn.click(timeout=10000)
        print("clicked Submit")
    except Exception as e:
        print("submit click err:", str(e)[:150])
        # fallback: press via dialog
        pg.screenshot(path='/tmp/xpuoj_submit_click_fail.png')
        b.close(); sys.exit(1)
    pg.wait_for_timeout(15000)

    body = pg.inner_text('body')
    m = re.search(r'/s/(\d+)', pg.url)
    sid = m.group(1) if m else None
    if not sid:
        m2 = re.search(r'#(\d+)', body)
        sid = m2.group(1) if m2 else None
    print("URL:", pg.url)
    print("SUBMISSION_ID:", sid or "unknown")
    # result hint
    for kw in ['Accepted','Compiling','Judging','Wrong Answer','Compile Error','Time Limit']:
        if kw.lower() in body.lower(): print("status keyword:", kw)
    pg.screenshot(path='/tmp/xpuoj_after_real_submit.png')
    b.close()
