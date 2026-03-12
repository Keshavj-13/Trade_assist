"""Lightweight service wrapper around the formal database helpers."""

from infra.database import get_open_positions as _get_open_positions, initialize_db


def init_db():
    initialize_db()


def get_open_positions(user_id=None):
    return _get_open_positions(user_id)
