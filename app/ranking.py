from datetime import datetime, timezone
import heapq
import math
from typing import Any, Dict, List

DEFAULT_WEIGHTS: Dict[str, float] = {
    "freshness": 2.0,
    "affinity": 1.5,
    "popularity": 1.0,
}


def parse_timestamp(ts: str) -> float:
    
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def compute_freshness(created_at_ts: float, now: float, half_life_hours: float = 24.0) -> float:
    
    age_hours = max(0.0, (now - created_at_ts) / 3600.0)
    return math.exp(-math.log(2) * (age_hours / half_life_hours))


def rank_feed(
    candidates: List[Dict[str, Any]],
    signals: Dict[int, Dict[str, float]],
    now: float,
    weights: Dict[str, float],
    limit: int,
) -> List[Dict[str, Any]]:
    
    if limit <= 0 or not candidates:
        return []

    w_fresh = float(weights.get("freshness", 0.0))
    w_aff = float(weights.get("affinity", 0.0))
    w_pop = float(weights.get("popularity", 0.0))

    heap: List[tuple] = []

    for cand in candidates:
        post_id = int(cand["post_id"])
        sig = signals.get(post_id, {})

        if "freshness" in sig:
            freshness = float(sig["freshness"])
        elif "created_at_ts" in sig:
            freshness = compute_freshness(float(sig["created_at_ts"]), now)
        elif "created_at" in cand and isinstance(cand["created_at"], str):
            freshness = compute_freshness(parse_timestamp(cand["created_at"]), now)
        else:
            freshness = 0.0

        affinity = float(sig.get("affinity", 0.0))
        popularity = float(sig.get("popularity", 0.0))

        score = round(
            w_fresh * freshness + w_aff * affinity + w_pop * popularity,
            6,
        )

       
        new_item = {
            "post_id": post_id,
            "user_id": cand["user_id"],
            "content": cand["content"],
            "created_at": cand["created_at"],
            "score": score,
        }

        
        entry = (score, -post_id, new_item)

        if len(heap) < limit:
            heapq.heappush(heap, entry)
        else:
            if (score, -post_id) > (heap[0][0], heap[0][1]):
                heapq.heappushpop(heap, entry)

    
    heap.sort(key=lambda x: (-x[0], -x[1]))
    return [item[2] for item in heap]