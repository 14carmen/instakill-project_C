"""Repository layer for database reads."""
from typing import Dict, List, Set
from app.db.database import get_connection


def user_exists(user_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM user WHERE id = ?", (user_id,)).fetchone()
        return row is not None


def get_all_posts() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, user_id, content, created_at FROM post"
        ).fetchall()
        return [dict(r) for r in rows]


def get_followed_ids(user_id: int) -> Set[int]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT followed_id FROM follow WHERE follower_id = ?", (user_id,)
        ).fetchall()
        return {r["followed_id"] for r in rows}


def get_liked_post_ids(user_id: int) -> Set[int]:
    with get_connection() as conn:
        rows = conn.execute(
            'SELECT post_id FROM "like" WHERE user_id = ?', (user_id,)
        ).fetchall()
        return {r["post_id"] for r in rows}


def get_post_like_counts() -> Dict[int, int]:
    with get_connection() as conn:
        rows = conn.execute(
            'SELECT post_id, COUNT(*) as cnt FROM "like" GROUP BY post_id'
        ).fetchall()
        return {r["post_id"]: r["cnt"] for r in rows}


def get_author_like_affinity(user_id: int) -> Dict[int, int]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.user_id AS author_id, COUNT(*) AS cnt
            FROM "like" l
            JOIN post p ON p.id = l.post_id
            WHERE l.user_id = ?
            GROUP BY p.user_id
            """,
            (user_id,),
        ).fetchall()
        return {r["author_id"]: r["cnt"] for r in rows}