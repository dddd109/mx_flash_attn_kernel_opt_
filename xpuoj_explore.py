"""XPUOJ auto-submit - step 1: login and explore submit page structure"""
from playwright.sync_api import sync_playwright
import time

USER = "muxi2026C2032@example.com"
PWD = "MDfm3jQ2eZsrVVAb"
# contest 11 problem 1 提交 URL (LibreOJ 格式)
SUBMIT_URL = "https://xpuoj.com/contest/11/problem/1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    ctx = browser.new_context()
    page = ctx.new_page()
    page.goto(SUBMIT_URL, timeout=40000)
    page.wait_for_timeout(3000)
    print("== 提交页 URL:", page.url)
    print("== 页面文本片段:", page.inner_text('body')[:300].replace('\n',' | '))
    # 找 Login 入口
    try:
        login_btn = page.locator('a:has-text("Login"), button:has-text("Login"), a:has-text("登录")').first
        if login_btn.count():
            print("== 找到 Login 按钮:", login_btn.inner_text())
            login_btn.click()
            page.wait_for_timeout(3000)
            print("== 登录页 URL:", page.url)
            # 打印登录表单字段
            inputs = page.locator('input').all()
            for i in inputs:
                print("== input:", i.get_attribute('name'), i.get_attribute('type'), i.get_attribute('placeholder'))
    except Exception as e:
        print("登录按钮处理失败:", str(e)[:200])
    browser.close()
