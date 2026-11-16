"""Application service: feed building and semantic search. No HTTP here; main.py maps errors to status codes."""
import math
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app import learned
from app.db import repository
from app.ranking import DEFAULT_WEIGHTS, compute_freshness, parse_timestamp, rank_feed
from app.retrieval import get_top_k_results
from app.search_index import search_index
from app.signals import SignalConfig, compute_features, genre_matrix


def search_posts_by_meaning(query: str, limit: int, all_posts_db: List[Dict]) -> List[Dict]:
    if not search_index.is_ready or not all_posts_db:
        return []

    query_emb = search_index.encode_query(query)
    top_results = get_top_k_results(query_emb, search_index.embeddings, search_index.post_ids, limit)

    posts_by_id = {post["id"]: post for post in all_posts_db}

    final_results = []
    for post_id, score in top_results:
        post_data = posts_by_id.get(post_id)
        if post_data:
            final_results.append({
                "post_id": post_id,
                "user_id": post_data["user_id"],
                "content": post_data["content"],
                "score": round(score, 4),
            })

    return final_results


class UserNotFoundError(Exception):
    """The feed was requested for a user that does not exist."""


class ModelUnavailableError(Exception):
    """strategy=learned was requested but like_model.npz is not loaded."""


Like = Tuple[int, int, float]          # (user_id, post_id, unix seconds)


def learned_features(
    user_id: int,
    posts: Sequence[Dict[str, Any]],
    likes: Sequence[Like],
    candidate_ids: Sequence[int],
    now: float,
    config: SignalConfig,
) -> np.ndarray:
    """The six notebook signals for `candidate_ids`, as a (C, 6) array. `likes` must all be before `now`."""
    index = {p["id"]: i for i, p in enumerate(posts)}
    created_ts = np.array([parse_timestamp(p["created_at"]) for p in posts], dtype=float)
    genres = genre_matrix(p.get("genres", "") for p in posts)

    known = [like for like in likes if like[1] in index]                # a like on a deleted post is ignored
    like_post_idx = np.array([index[post_id] for _, post_id, _ in known], dtype=int)
    like_ts = np.array([ts for _, _, ts in known], dtype=float)
    user_like_mask = np.array([u == user_id for u, _, _ in known], dtype=bool)

    return compute_features(
        now=now,
        created_ts=created_ts,
        genres=genres,
        like_post_idx=like_post_idx,
        like_ts=like_ts,
        user_like_mask=user_like_mask,
        candidate_idx=np.array([index[pid] for pid in candidate_ids], dtype=int),
        config=config,
    )


def build_user_feed(
    user_id: int,
    limit: int = 20,
    strategy: str = "for_you",
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """`now` defaults to the current time. A past `now` replays the feed as it was then:
    only posts created and likes given before `now` are seen."""
    if not repository.user_exists(user_id):
        raise UserNotFoundError(f"User {user_id} not found")
    if strategy == "learned" and not learned.is_available():
        raise ModelUnavailableError("Learned model not available")

    if now is None:
        now = time.time()

    all_posts = [p for p in repository.get_all_posts() if parse_timestamp(p["created_at"]) < now]
    likes: List[Like] = [
        (u, pid, ts) for u, pid, ts in
        ((u, pid, parse_timestamp(created)) for u, pid, created in repository.get_all_likes())
        if ts < now
    ]
    author_of = {p["id"]: p["user_id"] for p in all_posts}

    followed_ids = repository.get_followed_ids(user_id)
    liked_post_ids = {pid for u, pid, _ in likes if u == user_id}
    like_counts = Counter(pid for _, pid, _ in likes)
    author_likes = Counter(author_of[pid] for pid in liked_post_ids if pid in author_of)

    eligible_posts = [
        p for p in all_posts
        if p["user_id"] != user_id and p["id"] not in liked_post_ids
    ]

    if followed_ids:
        followed_candidates = [p for p in eligible_posts if p["user_id"] in followed_ids]
        raw_candidates = followed_candidates if followed_candidates else eligible_posts
    else:
        raw_candidates = eligible_posts

    candidates: List[Dict[str, Any]] = [
        {
            "post_id": p["id"],
            "user_id": p["user_id"],
            "content": p["content"],
            "created_at": p["created_at"],
        }
        for p in raw_candidates
    ]

    # Strategy: learned (the Week 5 network)
    if strategy == "learned":
        if not candidates:
            return {"user_id": user_id, "strategy": strategy, "results": []}
        features = learned_features(
            user_id, all_posts, likes, [c["post_id"] for c in candidates], now,
            SignalConfig.from_saved(learned.saved),
        )
        scores = learned.predict(features)
        post_ids = np.array([c["post_id"] for c in candidates])
        order = np.lexsort((post_ids, -scores))[:limit]                  # best score first, ties by ascending post_id
        return {
            "user_id": user_id,
            "strategy": strategy,
            "results": [{**candidates[i], "score": round(float(scores[i]), 6)} for i in order],
        }

    # Strategy: chronological
    if strategy == "chronological":
        ordered = sorted(
            candidates,
            key=lambda c: (-parse_timestamp(c["created_at"]), c["post_id"]),
        )
        return {
            "user_id": user_id,
            "strategy": strategy,
            "results": ordered[:limit],
        }

    # Strategy: for_you (Week 4 hand-tuned formula)
    signals: Dict[int, Dict[str, float]] = {}
    for c in candidates:
        pid = c["post_id"]
        author_id = c["user_id"]
        created_ts = parse_timestamp(c["created_at"])

        freshness = compute_freshness(created_ts, now)
        popularity = math.log1p(like_counts.get(pid, 0))
        follow_boost = 1.0 if author_id in followed_ids else 0.0
        prior_like_boost = 0.5 * math.log1p(author_likes.get(author_id, 0))
        affinity = follow_boost + prior_like_boost

        signals[pid] = {
            "freshness": freshness,
            "affinity": affinity,
            "popularity": popularity,
        }

    ranked = rank_feed(
        candidates=candidates,
        signals=signals,
        now=now,
        weights=DEFAULT_WEIGHTS,
        limit=limit,
    )

    return {
        "user_id": user_id,
        "strategy": strategy,
        "results": ranked,
    }
