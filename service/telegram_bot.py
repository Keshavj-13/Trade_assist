"""
Telegram bot integration and listener.
"""

import time
import requests
from infra.telegram import parse_command, send_message
from infra.logging import log
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def telegram_listener_loop():
    last_update_id = None
    poll_interval = 5
    log.info("Telegram listener started.")
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {"timeout": poll_interval}
            if last_update_id:
                params["offset"] = last_update_id + 1
            resp = requests.get(url, params=params, timeout=poll_interval + 5)
            if resp.status_code == 200:
                data = resp.json()
                for update in data.get("result", []):
                    last_update_id = update.get("update_id", last_update_id)
                    msg = update.get("message", {}).get("text")
                    if not isinstance(msg, str):
                        continue
                    try:
                        reply = parse_command(msg)
                        if reply:
                            send_message(reply)
                    except Exception as e:
                        log.error(f"Command handling error: {e}", exc_info=True)
            else:
                log.error(f"Telegram getUpdates error {resp.status_code}: {resp.text}")
        except Exception as e:
            log.error(f"Telegram polling error: {e}", exc_info=True)
        time.sleep(poll_interval)
