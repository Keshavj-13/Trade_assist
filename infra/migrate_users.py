"""Utility to migrate informal user state (JSON) into the formal database."""

import json
import os
from typing import Tuple

from config.settings import (
    DATA_DIR,
    DEFAULT_USER_INITIAL_CAPITAL,
    DEFAULT_USER_MAX_ALLOCATION,
    DEFAULT_USER_RISK_PER_TRADE,
    DEFAULT_USER_WALLET,
)
from infra.database import ensure_user_profile, initialize_db

USERS_FILE = os.path.join(DATA_DIR, "users.json")

RISK_PROFILE_MAP = {
    "balanced": (DEFAULT_USER_RISK_PER_TRADE, DEFAULT_USER_MAX_ALLOCATION),
    "conservative": (DEFAULT_USER_RISK_PER_TRADE * 0.5, DEFAULT_USER_MAX_ALLOCATION * 0.5),
    "aggressive": (min(0.5, DEFAULT_USER_RISK_PER_TRADE * 2), min(0.5, DEFAULT_USER_MAX_ALLOCATION * 1.5)),
}


def _resolve_profile(profile: str) -> Tuple[float, float]:
    if not profile:
        return DEFAULT_USER_RISK_PER_TRADE, DEFAULT_USER_MAX_ALLOCATION
    return RISK_PROFILE_MAP.get(profile.lower(), (DEFAULT_USER_RISK_PER_TRADE, DEFAULT_USER_MAX_ALLOCATION))


def _load_users():
    if not os.path.exists(USERS_FILE):
        print("No legacy users.json found; nothing to migrate.")
        return {}
    with open(USERS_FILE, "r") as fh:
        payload = json.load(fh)
    return payload.get("users", {})


def migrate_users():
    users = _load_users()
    if not users:
        print("No users defined in users.json.")
        return

    for username, details in users.items():
        wallet = float(details.get("wallet", DEFAULT_USER_WALLET))
        initial_capital = float(details.get("initial_capital", wallet)) if details.get("initial_capital") else wallet
        risk_per_trade, max_alloc = _resolve_profile(details.get("risk_profile", ""))
        password = details.get("password") or ""
        ensure_user_profile(
            username,
            wallet,
            initial_capital,
            risk_per_trade,
            max_alloc,
            password=password,
            notes="migrated from data/users.json",
        )
        print(f"Migrated user {username}: wallet={wallet} risk={risk_per_trade:.4f}")


def main():
    initialize_db()
    migrate_users()


if __name__ == "__main__":
    main()
