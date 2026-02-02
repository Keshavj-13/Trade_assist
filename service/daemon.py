"""
Long running loop placeholder for service.
"""

import time
from infra.logging import log
from infra.telegram import send_message, parse_command
from service.runner import run_once
import requests
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def daemon_loop():
    pass  # TODO: Implement daemon loop

def run_forever():
    last_update_id = None
    poll_interval = 10
    scan_interval = 300
    last_scan = 0
    log.info("Daemon started.")
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {"timeout": poll_interval}
            if last_update_id:
                params["offset"] = last_update_id + 1
            resp = requests.get(url, params=params, timeout=poll_interval+5)
            if resp.status_code == 200:
                data = resp.json()
                for update in data.get("result", []):
                    last_update_id = update["update_id"]
                    msg = update.get("message", {}).get("text", "")
                    if msg:
                        reply = parse_command(msg)
                        if reply:
                            send_message(reply)
            else:
                log.error(f"Telegram getUpdates error {resp.status_code}: {resp.text}")
        except Exception as e:
            log.error(f"Telegram polling error: {e}", exc_info=True)
        now = time.time()
        if now - last_scan > scan_interval:
            run_once()
            last_scan = now
        time.sleep(poll_interval)
