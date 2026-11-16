"""SQLite bootstrap for InstaKill. Uses the stdlib sqlite3 module (see R11)."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "instakill.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS post (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES user(id),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    genres TEXT NOT NULL DEFAULT ''          -- pipe-separated, e.g. "Comedy|Drama" (Week 5 affinity signal)
);

CREATE TABLE IF NOT EXISTS follow (
    follower_id INTEGER NOT NULL REFERENCES user(id),
    followed_id INTEGER NOT NULL REFERENCES user(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (follower_id, followed_id)
);

CREATE TABLE IF NOT EXISTS "like" (
    user_id INTEGER NOT NULL REFERENCES user(id),
    post_id INTEGER NOT NULL REFERENCES post(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, post_id)
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring a database created by an older week up to date (CREATE IF NOT EXISTS never adds columns)."""
    post_columns = {row["name"] for row in conn.execute("PRAGMA table_info(post)")}
    if "genres" not in post_columns:
        conn.execute("ALTER TABLE post ADD COLUMN genres TEXT NOT NULL DEFAULT ''")


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
