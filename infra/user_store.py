"""User and wallet helper backed by the formal SQLite schema."""

import hashlib

from config.settings import DEFAULT_USER_WALLET
from infra.database import (
    adjust_user_wallet as db_adjust_user_wallet,
    ensure_user_profile,
    fetch_user,
    get_wallet_balance,
    initialize_db,
    update_user_wallet as db_update_user_wallet,
)

initialize_db()


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def ensure_user(username: str, password: str = "") -> None:
    hashed = _hash(password) if password else None
    ensure_user_profile(username, password=hashed, wallet=DEFAULT_USER_WALLET)


def user_exists(username: str) -> bool:
    return bool(fetch_user(username))


def create_user(username: str, password: str, wallet: float = DEFAULT_USER_WALLET) -> None:
    hashed = _hash(password)
    ensure_user_profile(username, wallet=wallet, initial_capital=wallet, password=hashed)


def authenticate(username: str, password: str) -> bool:
    row = fetch_user(username)
    if not row:
        return False
    stored = row["password"] or ""
    if not password:
        return stored == ""
    return stored == _hash(password)


def get_wallet(username: str) -> float:
    return get_wallet_balance(username)


def update_wallet(username: str, amount: float) -> float:
    ensure_user(username)
    db_update_user_wallet(username, float(amount))
    return get_wallet(username)


def adjust_wallet(username: str, delta: float) -> float:
    ensure_user(username)
    new_wallet = db_adjust_user_wallet(username, delta)
    return float(new_wallet or 0.0)
