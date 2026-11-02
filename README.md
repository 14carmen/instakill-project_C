# InstaKill — Starter Repo

Mini social-media platform. Mandatory backend: Python + FastAPI + SQLite (default) / PostgreSQL (allowed). See the full brief and grading rubric in `masterplan.md` (course root).

What the instructor provides (this starter):

- `app/main.py` — FastAPI hello-world with `GET /health`
- `app/db/database.py` — SQLite bootstrap (creates tables on startup)
- `app/domain/models.py` — `User` / `Post` schema
- `scripts/seed_db.py` — 5-10 synthetic users/posts
- `tests/test_health.py` — test skeleton

What it does **not** provide: the feed, ranking, ML models, search. Those are the team's work, in the following weeks.

## Running it

```bash
pip install -e .
uvicorn app.main:app --reload
# other terminal:
curl localhost:8000/health

python scripts/seed_db.py
pytest
```

## Milestones (see masterplan.md for full details)

W1 Repo+API → W3 CRUD+DB → W4 Feed v0 → W7 Baseline+eval plan → W9 ML+pretrained feature → W11 Profile+optimize → W13 Feature freeze+E2E → W14 Red-team+Demo Day.

At the end of each week: a demonstrable commit/tag.


## Semantic Search Index
The in-memory semantic search index is built at application startup using FastAPI's `lifespan` event (in `app/main.py`). It loads all existing posts from the SQLite database, encodes them using `sentence-transformers`, and stores the embeddings in memory (`app/search_index.py`). 

To handle new posts in the future, the `POST /posts` endpoint would need to dynamically encode the incoming post's content and append the resulting embedding and `post_id` to the existing arrays in the `search_index` singleton.

```markdown
## Week 4 — Feed Ranking & Complexity

### Candidates and Cold Start
For a given `user_id`, the feed excludes the user's own posts and any posts the user has already liked. Candidate posts are selected from authors that the user follows. **Cold start behavior:** if a user follows nobody (or followed authors have no eligible posts), all eligible global posts are used as candidates.

### Ranking Signals & Formula
For `strategy=for_you`, each candidate post receives three signals:
1. **Freshness**: Exponential time decay $\exp(-\ln(2) \cdot \text{age\_hours} / 24)$.
2. **Affinity**: `1.0` if the user follows the author, plus `0.5 * log(1 + prior_likes_on_author)`.
3. **Popularity**: `log(1 + total_post_likes)`.

**Formula & Default Weights:**
$$\text{score} = 2.0 \cdot \text{freshness} + 1.5 \cdot \text{affinity} + 1.0 \cdot \text{popularity}$$
Equal scores(or equal timestamps in `chronological` mode) use ascending `post_id` as a deterministic tie-breaker.

### Complexity Note
For $C$ candidate posts and a limit of $k$, scoring and selecting the top-$k$ posts with `heapq` runs in **$O(C \log k)$** time and **$O(k)$** auxiliary memory. In a production system, the **popularity** signal (total like counts per post) and **author-affinity** counts would be precomputed first (via materialized counters or a cache updated on write), avoiding full `GROUP BY` aggregations on every `GET /feed` request.