import math
import sqlite3

import pytest

import app.db.database as database
import app.main
from app import service
from app.ranking import parse_timestamp
from scripts import evaluate

T = 100_000.0           # the cut-off of the fixture world
DAY = 86_400


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A tiny database with a past (before T) and a future (at or after T). Returns its path."""
    path = tmp_path / "world.db"
    monkeypatch.setattr(database, "DB_PATH", path)       # restored after the test, whatever evaluate.py does
    database.init_db()
    with sqlite3.connect(path) as conn:
        conn.executemany("INSERT INTO user (id, username) VALUES (?, ?)",
                         [(1, "ana"), (2, "bob"), (3, "cri"), (4, "dan"), (100, "catalogue")])
        posts = [(i, 100, f"post {i}", T - (20 - i) * 1_000, "Drama" if i % 2 else "Comedy") for i in range(1, 13)]
        posts += [(13, 100, "posted at T", T, "Drama"), (14, 100, "posted after T", T + 5_000, "Comedy")]
        conn.executemany("INSERT INTO post (id, user_id, content, created_at, genres) VALUES (?, ?, ?, ?, ?)", posts)
        likes = [
            (1, 1, T - 9_000), (1, 2, T - 8_000), (1, 3, T - 7_000),       # ana: 3 likes before T ...
            (1, 4, T + 1_000), (1, 5, T + 2_000), (1, 14, T + 6_000),     # ... 2 relevant after T (14 is not in the snapshot)
            (2, 1, T - 5_000), (2, 6, T + 1_000),                          # bob: only 1 like before T -> not evaluated
            (3, 1, T - 4_000), (3, 2, T - 3_000), (3, 6, T - 2_000), (3, 7, T - 1_000),
            (3, 13, T + 3_000),                                            # cri: the only later like is on a post of T -> not evaluated
            (4, 2, T - 6_000), (4, 3, T - 5_500), (4, 4, T - 5_000), (4, 8, T),   # dan: a like exactly at T is the future
            (2, 9, T - 100),
        ]
        conn.executemany('INSERT INTO "like" (user_id, post_id, created_at) VALUES (?, ?, ?)', likes)
        conn.executemany("INSERT INTO follow (follower_id, followed_id, created_at) VALUES (?, ?, ?)",
                         [(1, 100, T - 50_000), (4, 100, T), (2, 100, T + 10)])
    return path


def test_ndcg_on_hand_made_rankings():
    ranking = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert evaluate.ndcg_at_k(ranking, set()) == 0.0                                  # no relevant post
    assert evaluate.ndcg_at_k(ranking, {99}) == 0.0                                   # relevant, but not returned
    assert evaluate.ndcg_at_k(ranking, {10, 11, 12}) == pytest.approx(1.0)            # all relevant first
    assert evaluate.ndcg_at_k(ranking, {12}) == pytest.approx(1 / math.log2(4))       # one relevant post at position 3
    # two relevant posts at positions 1 and 4: (1 + 1/log2(5)) / (1 + 1/log2(3))
    assert evaluate.ndcg_at_k(ranking, {10, 13}) == pytest.approx((1 + 1 / math.log2(5)) / (1 + 1 / math.log2(3)))
    # more relevant posts than 10: the ideal top 10 is full, so 10 relevant returned = 1.0
    assert evaluate.ndcg_at_k(ranking, set(range(10, 40))) == pytest.approx(1.0)


def test_snapshot_has_no_future(world, tmp_path):
    snapshot = tmp_path / "snapshot.db"
    removed = evaluate.make_snapshot(world, snapshot, T)
    assert removed == {"post": 2, "like": 6, "follow": 2}
    with sqlite3.connect(snapshot) as conn:
        for table in ("post", '"like"', "follow"):
            times = [parse_timestamp(r[0]) for r in conn.execute(f"SELECT created_at FROM {table}")]
            assert times and max(times) < T, f"{table} has rows at or after T"
        orphan = conn.execute('SELECT COUNT(*) FROM "like" WHERE post_id NOT IN (SELECT id FROM post)').fetchone()[0]
        assert orphan == 0
    with sqlite3.connect(world) as conn:                                                # the original is untouched
        assert conn.execute('SELECT COUNT(*) FROM "like"').fetchone()[0] == 18


def test_protocol_users_and_relevance(world):
    rows = evaluate.run_protocol(world, T, seed=42, progress=False)["rows"]
    assert sorted({r["user_id"] for r in rows}) == [1, 4]                              # bob: too few likes; cri: no answer
    assert {r["strategy"] for r in rows} == set(evaluate.STRATEGIES)
    assert {r["user_id"]: r["n_relevant"] for r in rows} == {1: 2, 4: 1}               # post 14 is not in the snapshot
    assert all(r["user_group"] == "new (<10 likes)" for r in rows)
    assert all(0.0 <= r["ndcg_at_10"] <= 1.0 for r in rows)


def test_two_runs_give_identical_reports(world, tmp_path):
    fingerprint = evaluate.content_fingerprint(world)
    for out in ("run1", "run2"):
        evaluate.evaluate(world, "1970-01-02T03:46:40", 42, tmp_path / out, dashboard=False, progress=False)
    for name in ("eval_results.csv", "baseline_table.md"):
        assert (tmp_path / "run1" / name).read_bytes() == (tmp_path / "run2" / name).read_bytes()
    assert evaluate.content_fingerprint(world) == fingerprint                          # evaluating never changes the database
    assert database.DB_PATH == world                                                   # and points the app back at it


def test_dashboard_is_one_self_contained_file(world, tmp_path):
    pytest.importorskip("matplotlib")
    evaluate.evaluate(world, "1970-01-02T03:46:40", 42, tmp_path / "out", dashboard=True, progress=False)
    page = (tmp_path / "out" / "dashboard.html").read_text(encoding="utf-8")
    assert page.count("data:image/png;base64,") >= 3
    assert "<script" not in page and "http://" not in page and "https://" not in page


def test_uses_the_same_service_function_as_the_endpoint(world, monkeypatch):
    assert app.main.build_user_feed is service.build_user_feed                         # what GET /feed calls
    calls = []
    real = service.build_user_feed

    def counting(user_id, limit=20, strategy="for_you", now=None):
        calls.append((user_id, strategy, now))
        return real(user_id, limit=limit, strategy=strategy, now=now)

    monkeypatch.setattr(service, "build_user_feed", counting)
    rows = evaluate.run_protocol(world, T, seed=42, progress=False)["rows"]
    n_users = len({r["user_id"] for r in rows})
    assert len(calls) == 3 * n_users                                                   # chronological, for_you, learned
    assert {s for _, s, _ in calls} == set(evaluate.SERVICE_STRATEGIES)
    assert all(now == T for _, _, now in calls)
