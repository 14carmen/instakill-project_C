
import math
import time
import numpy as np
from app.learned import predict, saved
from typing import Any, Dict, List, Optional

from app.db import repository
from app.ranking import DEFAULT_WEIGHTS, compute_freshness, parse_timestamp, rank_feed
from app.retrieval import get_top_k_results
from app.search_index import search_index
from fastapi import HTTPException


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

    # STRATEGIA 1: Rețeaua Neuronală (Learned)
    if strategy == "learned":
        if not saved:
            raise HTTPException(status_code=503, detail="Learned model not available")
            
        DAY = 86400
        n_features = 6
        features = np.zeros((len(candidates), n_features))
        
        half_life_days = float(saved.get("half_life_days", 3 * 365))
        activity_cap = float(saved.get("activity_cap", 500))
        age_cap_days = float(saved.get("age_cap_days", 25 * 365))
        
        user_likes_count = len(liked_post_ids)
        activity_signal = min(1.0, np.log1p(user_likes_count) / np.log1p(activity_cap))
        
        max_lifetime = max(like_counts.values()) if like_counts else 1
        
        for i, c in enumerate(candidates):
            pid = c["post_id"]
            author_id = c["user_id"]
            created_ts = parse_timestamp(c["created_at"])
            age_days = max(0.0, (now - created_ts) / DAY)
            
            # 1. freshness
            freshness = 0.5 ** (age_days / half_life_days)
            # 2. affinity
            follow_boost = 1.0 if author_id in followed_ids else 0.0
            prior_like_boost = 0.5 * math.log1p(author_likes.get(author_id, 0))
            affinity = follow_boost + prior_like_boost
            # 3. popularity
            popularity = math.log1p(like_counts.get(pid, 0))
            # 4. activity
            activity = activity_signal
            # 5. lifetime
            lifetime = np.log1p(like_counts.get(pid, 0)) / np.log1p(max_lifetime)
            # 6. age
            age = min(1.0, age_days / age_cap_days)
            
            features[i] = [freshness, affinity, popularity, activity, lifetime, age]
        
        scores = predict(features)
        
        for i, c in enumerate(candidates):
            c["score"] = float(scores[i])
            
        ranked = sorted(candidates, key=lambda x: (-x["score"], x["post_id"]))[:limit]
        
        return {
            "user_id": user_id,
            "strategy": strategy,
            "results": ranked,
        }

    # STRATEGIA 2: Cronologic
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

    # STRATEGIA 3: For You (Manual)
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