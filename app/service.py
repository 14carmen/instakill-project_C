
import math
import time
from typing import Any, Dict, List, Optional

from app.db import repository
from app.ranking import DEFAULT_WEIGHTS, compute_freshness, parse_timestamp, rank_feed
from app.retrieval import get_top_k_results
from app.search_index import search_index


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
    """."""


def build_user_feed(
    user_id: int,
    limit: int = 20,
    strategy: str = "for_you",
    now: Optional[float] = None,
) -> Dict[str, Any]:
    
    if not repository.user_exists(user_id):
        raise UserNotFoundError(f"User {user_id} not found")

    if now is None:
        now = time.time()

    all_posts = repository.get_all_posts()
    followed_ids = repository.get_followed_ids(user_id)
    liked_post_ids = repository.get_liked_post_ids(user_id)
    like_counts = repository.get_post_like_counts()
    author_likes = repository.get_author_like_affinity(user_id)

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