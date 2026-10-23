from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite does not preserve tzinfo)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)