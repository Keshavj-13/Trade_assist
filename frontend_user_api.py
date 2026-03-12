"""Python helper for the Node UI to manage users and wallets."""

import argparse
import json
import sys

from infra.user_store import (
    authenticate,
    create_user,
    get_wallet,
    update_wallet,
    user_exists,
)


def _output(payload: dict) -> None:
    print(json.dumps(payload))


def _fail(message: str, exit_code: int = 1) -> None:
    sys.stderr.write(message)
    sys.exit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="User API for frontend control room")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser("login")
    login_parser.add_argument("--username", required=True)
    login_parser.add_argument("--password", default="")

    signup_parser = subparsers.add_parser("signup")
    signup_parser.add_argument("--username", required=True)
    signup_parser.add_argument("--password", required=True)
    signup_parser.add_argument("--wallet", type=float, default=0.0)

    wallet_parser = subparsers.add_parser("wallet")
    wallet_parser.add_argument("--username", required=True)
    wallet_parser.add_argument("--amount", type=float, required=True)

    args = parser.parse_args()

    if args.command == "login":
        if not user_exists(args.username):
            _fail("Unknown user", exit_code=2)
        if not authenticate(args.username, args.password):
            _fail("Invalid credentials", exit_code=3)
        wallet = get_wallet(args.username)
        _output({"wallet": wallet})
    elif args.command == "signup":
        if user_exists(args.username):
            _fail("Username exists", exit_code=4)
        create_user(args.username, args.password, args.wallet)
        _output({"wallet": float(args.wallet)})
    elif args.command == "wallet":
        if not user_exists(args.username):
            _fail("Unknown user", exit_code=2)
        wallet = update_wallet(args.username, args.amount)
        _output({"wallet": wallet})


if __name__ == "__main__":
    main()
