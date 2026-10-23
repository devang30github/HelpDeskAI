"""Create (or reset) an employee account.   python -m scripts.create_user alice --full-name "Alice Rao"
The password is read from a prompt so it never lands in shell history."""
import argparse
import getpass
import sys

from app.database.session import SessionLocal, init_db
from app.services.users import UserError, create_user


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a HelpDeskAI employee account")
    parser.add_argument("username")
    parser.add_argument("--full-name", required=True)
    parser.add_argument("--reset", action="store_true", help="update password/name if the user exists")
    args = parser.parse_args()

    init_db()
    password = getpass.getpass("Password (min 8 chars): ")
    if password != getpass.getpass("Confirm password: "):
        print("Passwords do not match.")
        return 1

    with SessionLocal() as db:
        try:
            user = create_user(db, args.username, password, args.full_name, reset=args.reset)
        except UserError as e:
            print(f"Error: {e}")
            return 1
    print(f"User '{user.username}' is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())