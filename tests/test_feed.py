"""Tests for Week 4 feed ranking and GET /feed endpoint."""
import copy
from fastapi.testclient import TestClient

from app.main import app
from app.ranking import rank_feed
from scripts.seed_db import seed


def test_rank_feed_ordering_and_determinism():
    candidates = [
        {"post_id": 1, "user_id": 2, "content": "post A", "created_at": "2026-10-25 10:00:00"},
        {"post_id": 2, "user_id": 3, "content": "post B", "created_at": "2026-10-25 11:00:00"},
        {"post_id": 3, "user_id": 4, "content": "post C", "created_at": "2026-10-25 12:00:00"},
    ]
    signals = {
        1: {"freshness": 0.2, "affinity": 1.0, "popularity": 0.5},
        2: {"freshness": 0.9, "affinity": 1.0, "popularity": 1.5},
        3: {"freshness": 0.5, "affinity": 0.0, "popularity": 0.1},
    }
    weights = {"freshness": 2.0, "affinity": 1.5, "popularity": 1.0}

    res1 = rank_feed(candidates, signals, now=1700000000.0, weights=weights, limit=2)
    res2 = rank_feed(candidates, signals, now=1700000000.0, weights=weights, limit=2)

    assert res1 == res2
    assert len(res1) == 2
    assert [r["post_id"] for r in res1] == [2, 1]
    assert res1[0]["score"] > res1[1]["score"]


def test_rank_feed_tie_breaker_ascending_post_id():
    candidates = [
        {"post_id": 10, "user_id": 2, "content": "ten", "created_at": "2026-10-25 10:00:00"},
        {"post_id": 3, "user_id": 2, "content": "three", "created_at": "2026-10-25 10:00:00"},
        {"post_id": 7, "user_id": 2, "content": "seven", "created_at": "2026-10-25 10:00:00"},
    ]
    signals = {
        10: {"freshness": 1.0, "affinity": 1.0, "popularity": 1.0},
        3: {"freshness": 1.0, "affinity": 1.0, "popularity": 1.0},
        7: {"freshness": 1.0, "affinity": 1.0, "popularity": 1.0},
    }
    weights = {"freshness": 1.0, "affinity": 1.0, "popularity": 1.0}

    res = rank_feed(candidates, signals, now=0.0, weights=weights, limit=2)
    assert [r["post_id"] for r in res] == [3, 7]


def test_rank_feed_does_not_mutate_inputs():
    candidates = [
        {"post_id": 1, "user_id": 2, "content": "post A", "created_at": "2026-10-25 10:00:00"}
    ]
    signals = {1: {"freshness": 0.8, "affinity": 1.0, "popularity": 0.5}}
    weights = {"freshness": 2.0, "affinity": 1.5, "popularity": 1.0}

    cand_copy = copy.deepcopy(candidates)
    sig_copy = copy.deepcopy(signals)
    weights_copy = copy.deepcopy(weights)

    rank_feed(candidates, signals, now=12345.0, weights=weights, limit=5)

    assert candidates == cand_copy
    assert signals == sig_copy
    assert weights == weights_copy


def test_feed_api_over_seeded_db():
    seed()
    client = TestClient(app)

    # 1. Test for_you pentru Alice (user_id=1)
    resp = client.get("/feed?user_id=1&limit=5&strategy=for_you")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == 1
    assert data["strategy"] == "for_you"
    assert len(data["results"]) > 0
    for post in data["results"]:
        assert post["user_id"] != 1  # Exclude postările proprii
        assert post["post_id"] != 2  # Exclude postarea 2 (deja apreciată de user 1)
        assert "score" in post

    # 2. Test cold start pentru Eve (user_id=5, nu urmărește pe nimeni)
    resp_cold = client.get("/feed?user_id=5&limit=5&strategy=chronological")
    assert resp_cold.status_code == 200
    cold_data = resp_cold.json()
    assert len(cold_data["results"]) > 0
    assert "score" not in cold_data["results"][0]

    # 3. Test user inexistent -> 404
    resp_404 = client.get("/feed?user_id=9999")
    assert resp_404.status_code == 404

    # 4. Test limit în afara intervalului -> 422 Client Error
    resp_bad = client.get("/feed?user_id=1&limit=100")
    assert resp_bad.status_code == 422