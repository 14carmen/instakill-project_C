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

import time


def create_user(username: str) -> Dict:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO user (username) VALUES (?)", 
            (username,)
        )
        return {"id": cursor.lastrowid, "username": username}

def get_user(user_id: int) -> Dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT id, username FROM user WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

def get_all_users() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT id, username FROM user").fetchall()
        return [dict(r) for r in rows]

def update_user(user_id: int, username: str) -> Dict | None:
    with get_connection() as conn:
        conn.execute("UPDATE user SET username = ? WHERE id = ?", (username, user_id))
        return get_user(user_id)

def delete_user(user_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM user WHERE id = ?", (user_id,))


def create_post(user_id: int, content: str) -> Dict:
    created_at = time.time()
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO post (user_id, content, created_at) VALUES (?, ?, ?)", 
            (user_id, content, created_at)
        )
        return {"id": cursor.lastrowid, "user_id": user_id, "content": content, "created_at": created_at}

def get_post(post_id: int) -> Dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT id, user_id, content, created_at FROM post WHERE id = ?", (post_id,)).fetchone()
        return dict(row) if row else None

def update_post(post_id: int, content: str) -> Dict | None:
    with get_connection() as conn:
        conn.execute("UPDATE post SET content = ? WHERE id = ?", (content, post_id))
        return get_post(post_id)

def delete_post(post_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM post WHERE id = ?", (post_id,))


def create_follow(follower_id: int, followed_id: int):
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO follow (follower_id, followed_id) VALUES (?, ?)", 
            (follower_id, followed_id)
        )

def delete_follow(follower_id: int, followed_id: int):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM follow WHERE follower_id = ? AND followed_id = ?", 
            (follower_id, followed_id)
        )

def get_following(user_id: int) -> List[int]:
    with get_connection() as conn:
        rows = conn.execute("SELECT followed_id FROM follow WHERE follower_id = ?", (user_id,)).fetchall()
        return [r["followed_id"] for r in rows]

def get_followers(user_id: int) -> List[int]:
    with get_connection() as conn:
        rows = conn.execute("SELECT follower_id FROM follow WHERE followed_id = ?", (user_id,)).fetchall()
        return [r["follower_id"] for r in rows]


def create_like(user_id: int, post_id: int, created_at: float):
    with get_connection() as conn:
        conn.execute(
            'INSERT OR IGNORE INTO "like" (user_id, post_id, created_at) VALUES (?, ?, ?)', 
            (user_id, post_id, created_at)
        )

def delete_like(user_id: int, post_id: int):
    with get_connection() as conn:
        conn.execute(
            'DELETE FROM "like" WHERE user_id = ? AND post_id = ?', 
            (user_id, post_id)
        )