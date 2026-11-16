"""Replay real likes through GET /feed's code and measure which strategy puts the liked post in the top 10.

For each sampled like (user u liked post p at time t), we ask build_user_feed for u's feed
as it was just before t (as_of = t) and look for p in the top K. Only the database is used:
run scripts/seed_movielens.py first.

Unlike the notebook's exam (1 liked item among 99 random ones), the liked post here competes
with EVERY post that existed at time t, so the numbers are much lower than in the notebook.
Compare the strategies with each other, not with the notebook.

Usage:  python scripts/evaluate_feed.py [moments]      (default 200, about 1-2 minutes)
"""
import math
import random
import sys
import time

from app.db import repository
from app.ranking import parse_timestamp
from app.service import build_user_feed

K = 10
MIN_HISTORY = 3                      # as in the notebook: the user already had 3+ likes
STRATEGIES = ("chronological", "for_you", "learned")


def sample_moments(n: int, last_fraction: float = 0.15, seed: int = 42):
    """Likes from the most recent `last_fraction` of the history (the notebook's test period)."""
    created = {p["id"]: parse_timestamp(p["created_at"]) for p in repository.get_all_posts()}
    likes = sorted((parse_timestamp(ts), u, pid) for u, pid, ts in repository.get_all_likes())
    start = likes[int(len(likes) * (1 - last_fraction))][0]

    seen_per_user: dict = {}
    eligible = []
    for ts, u, pid in likes:
        prior = seen_per_user.get(u, set())
        earlier = {p for p, t in prior if t < ts}
        if ts >= start and len(earlier) >= MIN_HISTORY and created.get(pid, math.inf) < ts:
            eligible.append((u, pid, ts))
        seen_per_user.setdefault(u, set()).add((pid, ts))
    random.Random(seed).shuffle(eligible)
    return eligible[:n]


def evaluate(n: int) -> None:
    moments = sample_moments(n)
    if not moments:
        sys.exit("No likes to replay: seed the database with scripts/seed_movielens.py first.")
    print(f"Replaying {len(moments)} real likes; the liked post competes with every post that existed then.\n")
    print(f"{'strategy':15s} {'HitRate@10':>10s} {'NDCG@10':>9s} {'ms/request':>11s}")
    for strategy in STRATEGIES:
        hits, ndcg, started = 0, 0.0, time.perf_counter()
        for user_id, post_id, ts in moments:
            feed = build_user_feed(user_id, limit=K, strategy=strategy, now=ts)
            ids = [r["post_id"] for r in feed["results"]]
            if post_id in ids:
                hits += 1
                ndcg += 1.0 / math.log2(ids.index(post_id) + 2)
        ms = 1000 * (time.perf_counter() - started) / len(moments)
        print(f"{strategy:15s} {hits / len(moments):10.3f} {ndcg / len(moments):9.4f} {ms:11.0f}")


if __name__ == "__main__":
    evaluate(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
