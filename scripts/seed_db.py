"""Seeds the InstaKill database with a handful of synthetic users and posts."""
"""
from app.db.database import get_connection, init_db

USERS = ["alice", "bob", "carol", "dave", "erin"]
POSTS = [
    (1, "hello world, this is my first post"),
    (1, "does anyone else think the feed is chronological only? boring"),
    (2, "just shipped a cosine similarity search, feels great"),
    (3, "spam spam spam buy now click here"),
    (4, "benchmarking my ranker against popularity baseline today"),
    (5, "red-team week is going to be brutal, good luck everyone"),
]


def seed() -> None:
    init_db()
    with get_connection() as conn:
        for username in USERS:
            conn.execute(
                "INSERT OR IGNORE INTO user (username) VALUES (?)", (username,)
            )
        for user_id, content in POSTS:
            conn.execute(
                "INSERT INTO post (user_id, content) VALUES (?, ?)", (user_id, content)
            )
        conn.commit()
    print(f"Seeded {len(USERS)} users and {len(POSTS)} posts.")


if __name__ == "__main__":
    seed()

    """

"""Seed script for InstaKill database."""
from app.db.database import get_connection, init_db

USERS = [
    (1, "alice", "2026-10-01 08:00:00"),
    (2, "bob", "2026-10-01 08:05:00"),
    (3, "charlie", "2026-10-01 08:10:00"),
    (4, "diana", "2026-10-01 08:15:00"),
    (5, "eve", "2026-10-01 08:20:00"),  # User 5 nu urmărește pe nimeni (cold start)
]

POSTS = [
    (1, 1, "hello world, this is my first post", "2026-10-25 10:00:00"),
    (2, 2, "just setting up my instakill account", "2026-10-25 11:00:00"),
    (3, 2, "working on the feed ranking algorithm today", "2026-10-25 14:00:00"),
    (4, 3, "spam spam spam buy now click here", "2026-10-25 12:00:00"),
    (5, 4, "benchmarking my ranker against popularity baseline today", "2026-10-25 15:00:00"),
    (6, 5, "red-team week is going to be brutal, good luck everyone", "2026-10-25 16:00:00"),
    (7, 3, "heapq makes top-k selection fast and memory efficient", "2026-10-25 17:00:00"),
    (8, 4, "freshness vs popularity is a classic recommender tradeoff", "2026-10-25 18:00:00"),
]

FOLLOWS = [
    (1, 2),
    (1, 3),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 4),
    (4, 1),
]

LIKES = [
    (1, 2),  # Alice a apreciat deja postarea 2 (va fi exclusă din feed-ul ei)
    (2, 1),
    (3, 1),
    (4, 1),
    (1, 5),
    (2, 5),
    (3, 5),
    (2, 3),
    (4, 3),
    (5, 3),
    (1, 7),
    (4, 7),
]


def seed() -> None:
    init_db()
    with get_connection() as conn:
        conn.execute('DELETE FROM "like"')
        conn.execute("DELETE FROM follow")
        conn.execute("DELETE FROM post")
        conn.execute("DELETE FROM user")

        conn.executemany(
            "INSERT INTO user (id, username, created_at) VALUES (?, ?, ?)",
            USERS,
        )
        conn.executemany(
            "INSERT INTO post (id, user_id, content, created_at) VALUES (?, ?, ?, ?)",
            POSTS,
        )
        conn.executemany(
            "INSERT INTO follow (follower_id, followed_id) VALUES (?, ?)",
            FOLLOWS,
        )
        conn.executemany(
            'INSERT INTO "like" (user_id, post_id) VALUES (?, ?)',
            LIKES,
        )
    print(
        f"Seeded {len(USERS)} users, {len(POSTS)} posts, "
        f"{len(FOLLOWS)} follows, and {len(LIKES)} likes."
    )


if __name__ == "__main__":
    seed()