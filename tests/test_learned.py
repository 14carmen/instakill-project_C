"""Week 5: the learned feed must see exactly the signals the notebook trained on (no train/serve skew)."""
import csv
import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.db.database as database
from app import learned
from app.db import repository
from app.main import app
from app.ranking import parse_timestamp
from app.service import learned_features
from app.signals import DAY, FEATURES, SignalConfig, compute_features, genre_matrix

CONFIG = SignalConfig()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    return tmp_path


def test_model_file_matches_the_notebook():
    """The three golden rows printed by the export cell of week05.ipynb."""
    golden = np.array([
        [0.020159, 0.15, 0.0, 0.222999, 0.518748, 0.675891],
        [0.387413, 0.10, 0.0, 0.222999, 0.000000, 0.164167],
        [0.023264, 0.10, 0.0, 0.222999, 0.459336, 0.651094],
    ])
    assert learned.is_available()
    assert tuple(learned.saved["features"]) == FEATURES
    assert SignalConfig.from_saved(learned.saved) == CONFIG
    assert np.allclose(learned.predict(golden), [0.480419, 0.009535, 0.381069], atol=2e-6)


def test_signals_match_notebook_self_check_1():
    """Self-check 1 of week05.ipynb, on compute_features: same tiny world, same expected numbers."""
    now = CONFIG.half_life_days * DAY                           # exactly one half-life after the items appeared
    genres = genre_matrix(["Drama", "Drama", "Drama", "Comedy", "Drama", "Drama|Comedy", "Drama"])
    created = np.zeros(7)
    # (user, item, time): user 8 liked item 6 more than a year ago; user 7 liked items 0-3; users 8, 9 liked item 4
    likes = [(8, 6, now - 400 * DAY)] + [(7, r, now - 10 * DAY) for r in range(4)] + [(8, 4, now - 10 * DAY), (9, 4, now - 5 * DAY)]
    post_idx = np.array([l[1] for l in likes])
    like_ts = np.array([l[2] for l in likes])
    users = np.array([l[0] for l in likes])

    def features(user, rows):
        return compute_features(now=now, created_ts=created, genres=genres, like_post_idx=post_idx, like_ts=like_ts,
                                user_like_mask=users == user, candidate_idx=np.array(rows), config=CONFIG)

    f = features(7, [4, 5, 0])
    assert abs(f[0, 0] - 0.5) < 1e-9                                         # freshness after one half-life
    assert abs(f[0, 1] - 0.75) < 1e-9 and abs(f[1, 1] - 0.5) < 1e-9          # affinity 3/(4*1) and (3+1)/(4*2)
    assert abs(f[0, 2] - 1.0) < 1e-9 and abs(f[2, 2] - 0.5) < 1e-9           # popularity: recent likes / max
    assert abs(f[0, 3] - np.log1p(4) / np.log1p(500)) < 1e-9                 # activity of a user with 4 likes
    assert abs(f[0, 4] - 1.0) < 1e-9                                         # lifetime of the most-liked item
    assert abs(f[0, 5] - CONFIG.half_life_days / (25 * 365)) < 1e-9          # age is linear
    assert features(3, [4])[0, 1] == 0.0                                     # no likes -> affinity 0
    g = features(7, [6])[0]
    assert g[2] == 0.0 and g[4] > 0.0                                        # old like: out of popularity, still in lifetime


def _tiny_movielens_zip(path):
    movies = [("movieId", "title", "genres"), (10, "Old Drama", "Drama"), (20, "Comedy", "Comedy"),
              (30, "Drama Comedy", "Drama|Comedy"), (40, "No genre", "(no genres listed)"), (50, "Never rated", "Drama")]
    ratings = [("userId", "movieId", "rating", "timestamp"),
               (1, 10, 4.0, 1000), (1, 20, 3.5, 900),       # movie 20 first seen at 900 by a rating that is not a like
               (2, 10, 5.0, 2000), (2, 30, 4.5, 3000), (3, 40, 2.0, 2500), (3, 20, 4.0, 4000)]
    buffers = {}
    for name, rows in (("movies.csv", movies), ("ratings.csv", ratings)):
        out = io.StringIO()
        csv.writer(out).writerows(rows)
        buffers[name] = out.getvalue()
    with zipfile.ZipFile(path, "w") as z:
        for name, text in buffers.items():
            z.writestr(f"ml-latest-small/{name}", text)
    return path


def test_seed_movielens_keeps_ids_times_and_genres(temp_db):
    from scripts.seed_movielens import CATALOGUE_USER_ID, seed

    counts = seed(str(_tiny_movielens_zip(temp_db / "ml.zip")))
    assert counts == {"users": 4, "posts": 4, "likes": 4}                     # movie 50 was never rated: no post

    posts = {p["id"]: p for p in repository.get_all_posts()}
    assert set(posts) == {10, 20, 30, 40}                                    # post id == movieId
    assert parse_timestamp(posts[20]["created_at"]) == 900                   # first rating of any value
    assert posts[30]["genres"] == "Drama|Comedy" and posts[40]["genres"] == ""
    assert all(p["user_id"] == CATALOGUE_USER_ID for p in posts.values())

    likes = {(u, p) for u, p, _ in repository.get_all_likes()}
    assert likes == {(1, 10), (2, 10), (2, 30), (3, 20)}                     # rating >= 4 only
    assert repository.get_user(2)["username"] == "ML_User_2"


def test_learned_signals_after_seeding_match_the_formulas(temp_db):
    from scripts.seed_movielens import seed

    seed(str(_tiny_movielens_zip(temp_db / "ml.zip")))
    now = 3500.0
    posts = [p for p in repository.get_all_posts() if parse_timestamp(p["created_at"]) < now]
    likes = [(u, p, parse_timestamp(t)) for u, p, t in repository.get_all_likes() if parse_timestamp(t) < now]
    f = learned_features(2, posts, likes, [20, 30], now, CONFIG)             # user 2 liked a Drama and a Drama|Comedy

    user_genres = np.array([2.0, 1.0])                                       # Drama x2, Comedy x1 -> total 3
    assert np.isclose(f[0, 1], user_genres[1] / (3 * 1))                     # Comedy
    assert np.isclose(f[1, 1], (user_genres[0] + user_genres[1]) / (3 * 2))  # Drama|Comedy
    assert np.isclose(f[0, 3], np.log1p(2) / np.log1p(500))                  # two likes before `now`
    assert np.isclose(f[1, 4], np.log1p(1) / np.log1p(2))                    # movie 10 has the most likes (2)
    assert f[0, 4] == 0.0                                                    # movie 20's like comes after `now`


def test_learned_feed_api(temp_db):
    from scripts.seed_movielens import seed

    seed(str(_tiny_movielens_zip(temp_db / "ml.zip")))
    client = TestClient(app)

    resp = client.get("/feed", params={"user_id": 1, "strategy": "learned", "limit": 10, "as_of": 5000})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["post_id"] for r in results] and 10 not in [r["post_id"] for r in results]   # already liked
    scores = [r["score"] for r in results]
    assert all(0.0 <= s <= 1.0 for s in scores) and scores == sorted(scores, reverse=True)

    replay = client.get("/feed", params={"user_id": 1, "strategy": "learned", "as_of": 950}).json()["results"]
    assert [r["post_id"] for r in replay] == [20]                           # only movie 20 existed before t=950


def test_learned_feed_without_model_is_503(temp_db, monkeypatch):
    database.init_db()
    repository.create_user("solo")
    monkeypatch.setattr(learned, "saved", {})
    resp = TestClient(app).get("/feed", params={"user_id": 1, "strategy": "learned"})
    assert resp.status_code == 503
