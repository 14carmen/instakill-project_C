"""Seed InstaKill with MovieLens latest-small, written directly into SQLite.

Same proxy as the Week 5 notebook, so the learned model sees the data it was trained on:

| MovieLens                                  | InstaKill                         |
|--------------------------------------------|-----------------------------------|
| movie (only movies somebody rated)         | post, id = movieId                |
| the first time anybody rated the movie     | post.created_at (unix seconds)    |
| the movie's genres                         | post.genres ("Comedy|Drama")      |
| user                                       | user, id = userId                 |
| a rating of 4 or more                      | a like, created_at = its timestamp |

All posts belong to one catalogue account (CATALOGUE_USER_ID). The script replaces the
contents of the database, so it is deterministic: run it twice, get the same tables.

Usage:  python scripts/seed_movielens.py path/to/ml-latest-small.zip
"""
import csv
import io
import sys
import zipfile
from typing import Dict, List, Tuple

from app.db.database import get_connection, init_db

LIKE_THRESHOLD = 4.0
CATALOGUE_USER_ID = 1_000_000          # outside MovieLens' user ids (1..610)
CATALOGUE_USERNAME = "movielens_catalogue"
NO_GENRES = "(no genres listed)"


def _read_csv(archive: zipfile.ZipFile, suffix: str) -> List[Dict[str, str]]:
    name = next(n for n in archive.namelist() if n.endswith(suffix))
    with archive.open(name) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline="")))


def load_movielens(zip_path: str) -> Tuple[list, list, list]:
    """Returns (users, posts, likes) rows ready for INSERT."""
    with zipfile.ZipFile(zip_path) as archive:
        movies = {int(r["movieId"]): (r["title"], r["genres"]) for r in _read_csv(archive, "movies.csv")}
        ratings = _read_csv(archive, "ratings.csv")

    first_seen: Dict[int, int] = {}
    user_ids = set()
    likes = []
    for r in ratings:
        user_id, movie_id, ts = int(r["userId"]), int(r["movieId"]), int(r["timestamp"])
        first_seen[movie_id] = min(ts, first_seen.get(movie_id, ts))      # every rating counts, not only likes
        user_ids.add(user_id)
        if float(r["rating"]) >= LIKE_THRESHOLD:
            likes.append((user_id, movie_id, ts))

    users = [(CATALOGUE_USER_ID, CATALOGUE_USERNAME)] + [(u, f"ML_User_{u}") for u in sorted(user_ids)]
    posts = []
    for movie_id in sorted(first_seen):
        title, genres = movies.get(movie_id, ("?", ""))
        clean_genres = "|".join(g for g in genres.split("|") if g and g != NO_GENRES)
        posts.append((movie_id, CATALOGUE_USER_ID, title, first_seen[movie_id], clean_genres))
    likes.sort(key=lambda like: (like[2], like[0], like[1]))
    return users, posts, likes


def seed(zip_path: str) -> Dict[str, int]:
    users, posts, likes = load_movielens(zip_path)
    init_db()
    with get_connection() as conn:
        for table in ('"like"', "follow", "post", "user"):
            conn.execute(f"DELETE FROM {table}")
        conn.executemany("INSERT INTO user (id, username) VALUES (?, ?)", users)
        conn.executemany("INSERT INTO post (id, user_id, content, created_at, genres) VALUES (?, ?, ?, ?, ?)", posts)
        conn.executemany('INSERT INTO "like" (user_id, post_id, created_at) VALUES (?, ?, ?)', likes)
    counts = {"users": len(users), "posts": len(posts), "likes": len(likes)}
    print(f"Seeded {counts['users']:,} users (incl. the catalogue account), {counts['posts']:,} posts, {counts['likes']:,} likes.")
    return counts


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/seed_movielens.py <path_to_ml-latest-small.zip>")
        sys.exit(1)
    seed(sys.argv[1])
