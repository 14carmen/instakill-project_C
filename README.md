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

python scripts/seed_db.py                                 # 5 hand-made users, for the Week 4 tests
python scripts/seed_movielens.py ml-latest-small.zip     # or: the MovieLens proxy, for strategy=learned
pytest
```

## Milestones (see masterplan.md for full details)

W1 Repo+API → W3 CRUD+DB → W4 Feed v0 → W7 Baseline+eval plan → W9 ML+pretrained feature → W11 Profile+optimize → W13 Feature freeze+E2E → W14 Red-team+Demo Day.

At the end of each week: a demonstrable commit/tag.


## Semantic Search Index
The in-memory semantic search index is built at application startup using FastAPI's `lifespan` event (in `app/main.py`). It loads all existing posts from the SQLite database, encodes them using `sentence-transformers`, and stores the embeddings in memory (`app/search_index.py`). 

To handle new posts in the future, the `POST /posts` endpoint would need to dynamically encode the incoming post's content and append the resulting embedding and `post_id` to the existing arrays in the `search_index` singleton.

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

## Week 5 — Learned feed (`strategy=learned`)

`GET /feed?strategy=learned` ranks the candidates with the one-hidden-layer network trained in `week05.ipynb` (6 → 32 tanh → 1 sigmoid, 257 parameters, val NDCG@10 0.4545 / test 0.4646 vs 0.4267 for the best straight line). The trained weights, the standardisation (`mu`, `sd`), the signal order and the signal constants are in `app/like_model.npz`; `app/learned.py` loads it once and `predict` is the notebook's code.

### Data: `scripts/seed_movielens.py`
Download `ml-latest-small.zip` from <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip> (it is not committed) and run `python scripts/seed_movielens.py ml-latest-small.zip`. The script writes straight into SQLite and **replaces** the database contents. It uses the notebook's proxy: a post is a movie (`post.id = movieId`) created the first time anybody rated it, with its genres in `post.genres`; a user keeps their MovieLens `userId`; a like is a rating of 4 or more, at the rating's time. All posts belong to one `movielens_catalogue` account. Result: 611 users, 9,724 posts, 48,580 likes.

### Signals: `app/signals.py`
The six signals are computed exactly like `LiveSignals.features` in the notebook (same formulas, same constants read from the `.npz`), vectorised with NumPy, from the likes strictly before `now`:

| signal | formula |
|---|---|
| freshness | `0.5 ** (age_days / 1095)` |
| affinity | genre overlap: `(post_genres · user_genre_counts) / (user_genre_total × n_genres(post))` |
| popularity | likes in the last 365 days ÷ the maximum of that over all posts |
| activity | `min(1, log1p(user likes) / log1p(500))` |
| lifetime | `log1p(likes ever) ÷ log1p(max likes ever)` |
| age | `min(1, age_days / 9125)` |

This is deliberately **not** the Week 4 affinity/popularity: feeding the network other formulas would be train/serve skew. `tests/test_learned.py` guards it with the notebook's golden rows and self-check 1, and checks the seeder end to end. On the real data, the signals were compared with the notebook's `LiveSignals` on 37 real moments × 100 candidates: identical.

Posts without genres (the synthetic ones, or new posts from `POST /posts`) get affinity 0, as a user with no history does in the notebook.

### Time and `as_of`
MovieLens likes stop in 2018, so at today's date every movie is "old" (age capped at 1) and nobody liked anything in the last year (popularity 0). `as_of=<unix time>` replays the feed at a past moment, which is how the notebook evaluated the model, e.g. `GET /feed?user_id=514&strategy=learned&as_of=1454621693`.

### Cost
One request loads the posts and likes, builds the signals in `O(P·G + L + C)` (P posts, G = 19 genres, L likes, C candidates) and runs the network in `O(C·6·H)` with H = 32. On the full MovieLens seed this takes about 150 ms on a laptop, dominated by reading and parsing the rows; the forward pass itself takes about 8 ms for 10k candidates (notebook timing). In production the per-post counters (likes ever, likes in the last year) and the per-user genre counts would be maintained on write instead of recomputed per request.

