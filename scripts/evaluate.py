"""Offline evaluation of the four feed strategies on our own database (Week 7).

Question: on our own data, which feed strategy puts the posts a user is going to like into
their top 10, by how much, for which users, and how sure are we?

Protocol (details in docs/evaluation-protocol.md):
1. Pick a cut-off T (default: the moment by which 80 % of the likes had happened).
2. Snapshot: copy the database and delete every like, post and follow at or after T.
3. Users: at least 3 likes before T and at least 1 like after T on a post of the snapshot.
4. For each user and strategy, ask app.service.build_user_feed (the function GET /feed calls)
   for the top 10 on the snapshot, with now = T. `popularity` is not an API strategy: it is a
   scorer of this script only, over the candidates the service returns.
5. Relevant posts = the snapshot posts the user liked after T. Metric: NDCG@10.
6. 95 % paired bootstrap over users (fixed seed): each strategy's mean, and learned minus the
   best baseline, overall and per slice.

Usage:
    python scripts/evaluate.py [--db instakill.db] [--cutoff 2016-02-04T21:34:50] [--seed 42] [--out reports/]

The same database and arguments give byte-identical eval_results.csv and baseline_table.md.
Needs the optional extra for the dashboard: pip install -e ".[eval]"
"""
import argparse
import base64
import csv
import hashlib
import html
import io
import math
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.db.database as database                      # noqa: E402  (after the sys.path fix)
from app import service                                  # noqa: E402
from app.ranking import parse_timestamp                  # noqa: E402

K = 10
CUTOFF_QUANTILE = 0.80
MIN_LIKES_BEFORE = 3
N_BOOT = 2000
SMALL_SLICE = 20                                         # fewer users than this: read the interval, not the verdict
POPULARITY_DAYS = 365
DAY = 86_400
SERVICE_STRATEGIES = ("chronological", "for_you", "learned")
STRATEGIES = ("chronological", "popularity", "for_you", "learned")
BASELINES = ("chronological", "popularity", "for_you")
COLORS = {"chronological": "#9AA6BD", "popularity": "#F7A30F", "for_you": "#10ADB6", "learned": "#0B2F63"}
USER_GROUPS = ("new (<10 likes)", "casual (10-49)", "regular (50-199)", "power (200+)")
TASTE_GROUPS = ("mostly new posts (<1 year)", "mostly older posts")

# The time cuts of the Week 5 lab (quantiles 0.70 and 0.85 of the MovieLens like times):
# the model was trained on likes before LAB_T_VAL and tuned on likes in [LAB_T_VAL, LAB_T_TEST).
LAB_T_VAL = 1_363_686_948          # 2013-03-19 09:55:48 UTC
LAB_T_TEST = 1_468_118_189         # 2016-07-10 02:36:29 UTC


# ---------------------------------------------------------------- the metric and the intervals

def ndcg_at_k(ranked_ids: Sequence[int], relevant: set, k: int = K) -> float:
    """DCG = sum(rel_i / log2(i + 2)) over the first k positions, divided by the DCG of a perfect top k."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(list(ranked_ids)[:k]) if pid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def paired_bootstrap(a, b, seed: int, n_boot: int = N_BOOT) -> Dict[str, float]:
    """95 % interval of mean(a - b), resampling users (the same user for a and b)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"diff": float(d.mean()), "low": float(low), "high": float(high)}


def verdict(r: Dict[str, float]) -> str:
    return "wins" if r["low"] > 0 else "loses" if r["high"] < 0 else "unclear"


def user_group(likes_before: int) -> str:
    if likes_before < 10:
        return USER_GROUPS[0]
    if likes_before < 50:
        return USER_GROUPS[1]
    if likes_before < 200:
        return USER_GROUPS[2]
    return USER_GROUPS[3]


# ---------------------------------------------------------------- database: cut-off, snapshot, fingerprint

def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def default_cutoff(db_path: Path) -> float:
    """The moment by which CUTOFF_QUANTILE of the likes had happened."""
    with _connect(db_path) as conn:
        times = sorted(parse_timestamp(r["created_at"]) for r in conn.execute('SELECT created_at FROM "like"'))
    if not times:
        raise SystemExit("The database has no likes: seed it first (python scripts/seed_movielens.py ml-latest-small.zip).")
    return times[int(len(times) * CUTOFF_QUANTILE)]


def parse_cutoff(text: str) -> float:
    """An ISO date ('2016-02-04', '2016-02-04T21:34:50'), UTC when no time zone is given."""
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def content_fingerprint(db_path: Path) -> str:
    """sha256 of the rows (not of the file, whose bytes change with free pages). User sign-up times are
    left out: the seeder stamps them with the time it ran."""
    queries = [
        "SELECT id, username FROM user ORDER BY id",
        "SELECT id, user_id, content, created_at, genres FROM post ORDER BY id",
        'SELECT user_id, post_id, created_at FROM "like" ORDER BY user_id, post_id',
        "SELECT follower_id, followed_id, created_at FROM follow ORDER BY follower_id, followed_id",
    ]
    h = hashlib.sha256()
    with _connect(db_path) as conn:
        for query in queries:
            for row in conn.execute(query):
                h.update(("\x1f".join(str(v) for v in tuple(row)) + "\n").encode("utf-8"))
            h.update(b"\x1e")
    return h.hexdigest()


def make_snapshot(source: Path, target: Path, cutoff: float) -> Dict[str, int]:
    """Copy `source` to `target`, then delete from the copy every like, post and follow at or after `cutoff`.
    Times are stored in mixed formats (unix seconds, ISO dates), so they are compared in Python."""
    database.DB_PATH = target                                 # init_db() migrates the copy, never the original
    src, dst = sqlite3.connect(source), sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    database.init_db()
    removed = {}
    with _connect(target) as conn:
        for table in ("post", '"like"', "follow"):
            late = [r["rid"] for r in conn.execute(f"SELECT rowid AS rid, created_at FROM {table}")
                    if parse_timestamp(r["created_at"]) >= cutoff]
            conn.executemany(f"DELETE FROM {table} WHERE rowid = ?", [(r,) for r in late])
            removed[table.strip('"')] = len(late)
        orphans = conn.execute('DELETE FROM "like" WHERE post_id NOT IN (SELECT id FROM post)').rowcount
        removed["like"] += orphans
    conn.close()
    return removed


def code_version() -> str:
    """The last commit that changed the evaluated code. Committing the reports does not change it,
    so a fresh clone reproduces baseline_table.md byte for byte."""
    paths = ["app", "scripts", "pyproject.toml"]
    try:
        commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", *paths], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"
    return (commit or "uncommitted") + (" + uncommitted changes" if dirty else "")


# ---------------------------------------------------------------- the protocol

def run_protocol(db_path: Path, cutoff: float, seed: int, progress: bool = True) -> Dict:
    """Build the snapshot, ask every strategy, score. Returns the per-user rows and the run metadata."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="instakill-eval-"))
    snapshot = tmp_dir / "snapshot.db"
    original_path = database.DB_PATH
    try:
        removed = make_snapshot(db_path, snapshot, cutoff)

        with _connect(db_path) as conn:                              # the answers: likes after T ...
            all_likes = [(r["user_id"], r["post_id"], parse_timestamp(r["created_at"]))
                         for r in conn.execute('SELECT user_id, post_id, created_at FROM "like"')]
        conn.close()
        with _connect(snapshot) as conn:                             # ... on posts that exist in the snapshot
            snapshot_posts = {r["id"]: parse_timestamp(r["created_at"]) for r in conn.execute("SELECT id, created_at FROM post")}
            n_snapshot_likes = conn.execute('SELECT COUNT(*) FROM "like"').fetchone()[0]
        conn.close()

        before: Dict[int, int] = {}
        after: Dict[int, set] = {}
        after_times: Dict[int, List[float]] = {}
        recent: Dict[int, int] = {}
        for user_id, post_id, ts in all_likes:
            if ts < cutoff:
                before[user_id] = before.get(user_id, 0) + 1
                if ts >= cutoff - POPULARITY_DAYS * DAY:
                    recent[post_id] = recent.get(post_id, 0) + 1
            elif post_id in snapshot_posts:
                after.setdefault(user_id, set()).add(post_id)
                after_times.setdefault(user_id, []).append(ts)
        users = sorted(u for u in after if before.get(u, 0) >= MIN_LIKES_BEFORE)
        if not users:
            raise SystemExit("No user has 3+ likes before the cut-off and 1+ like after it: pick another --cutoff.")

        rows = []
        for n, user_id in enumerate(users, 1):
            if progress and (n % 25 == 0 or n == len(users)):
                print(f"  user {n}/{len(users)}", file=sys.stderr)
            relevant = after[user_id]
            feeds = {}
            everything = service.build_user_feed(user_id, limit=10**9, strategy="chronological", now=cutoff)["results"]
            feeds["chronological"] = [r["post_id"] for r in everything[:K]]
            candidates = sorted((r["post_id"] for r in everything), key=lambda pid: (-recent.get(pid, 0), pid))
            feeds["popularity"] = candidates[:K]                                 # scorer of this script only
            for strategy in ("for_you", "learned"):
                feeds[strategy] = [r["post_id"] for r in
                                   service.build_user_feed(user_id, limit=K, strategy=strategy, now=cutoff)["results"]]
            new_share = sum(cutoff - snapshot_posts[p] < POPULARITY_DAYS * DAY for p in relevant) / len(relevant)
            for strategy in STRATEGIES:
                rows.append({
                    "user_id": user_id,
                    "strategy": strategy,
                    "ndcg_at_10": ndcg_at_k(feeds[strategy], relevant),
                    "n_relevant": len(relevant),
                    "user_group": user_group(before[user_id]),
                    "taste": TASTE_GROUPS[0] if new_share >= 0.5 else TASTE_GROUPS[1],
                })
    finally:
        database.DB_PATH = original_path
        shutil.rmtree(tmp_dir, ignore_errors=True)

    after_arr = np.array(sorted(ts for u in users for ts in after_times[u]))
    meta = {
        "cutoff": cutoff,
        "seed": seed,
        "n_users": len(users),
        "n_relevant": int(sum(len(after[u]) for u in users)),
        "fingerprint": content_fingerprint(db_path),
        "code": code_version(),
        "snapshot_likes": n_snapshot_likes,
        "snapshot_posts": len(snapshot_posts),
        "removed": removed,
        "after_in_lab_train": int((after_arr < LAB_T_VAL).sum()),
        "after_in_lab_val": int(((after_arr >= LAB_T_VAL) & (after_arr < LAB_T_TEST)).sum()),
        "after_in_lab_test": int((after_arr >= LAB_T_TEST).sum()),
    }
    return {"rows": rows, "meta": meta}


# ---------------------------------------------------------------- summaries

def _scores(rows, users: Sequence[int], strategy: str) -> np.ndarray:
    by_user = {r["user_id"]: r["ndcg_at_10"] for r in rows if r["strategy"] == strategy}
    return np.array([by_user[u] for u in users])


def summarize(rows: List[Dict], seed: int) -> Dict:
    users = sorted({r["user_id"] for r in rows})
    info = {r["user_id"]: r for r in rows if r["strategy"] == "learned"}
    scores = {s: _scores(rows, users, s) for s in STRATEGIES}

    overall = {s: {**paired_bootstrap(scores[s], np.zeros(len(users)), seed), "n": len(users)} for s in STRATEGIES}

    def lift(selected: List[int], label: str, column: str) -> Optional[Dict]:
        if not selected:
            return None
        idx = np.array([users.index(u) for u in selected])
        best = max(BASELINES, key=lambda s: (scores[s][idx].mean(), -BASELINES.index(s)))
        r = paired_bootstrap(scores["learned"][idx], scores[best][idx], seed)
        return {"column": column, "slice": label, "n": len(selected), "best_baseline": best,
                "learned": float(scores["learned"][idx].mean()), "baseline": float(scores[best][idx].mean()),
                **r, "verdict": verdict(r)}

    lifts = [lift(users, "all users", "overall")]
    group_means = {}
    for column, groups in (("user_group", USER_GROUPS), ("taste", TASTE_GROUPS)):
        for g in groups:
            selected = [u for u in users if info[u][column] == g]
            row = lift(selected, g, column)
            if row:
                lifts.append(row)
            if column == "user_group" and selected:
                idx = np.array([users.index(u) for u in selected])
                group_means[g] = {s: {**paired_bootstrap(scores[s][idx], np.zeros(len(idx)), seed), "n": len(idx)}
                                  for s in STRATEGIES}
    return {"overall": overall, "lifts": lifts, "group_means": group_means}


# ---------------------------------------------------------------- reports

def write_csv(rows: List[Dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["user_id", "strategy", "ndcg_at_10", "n_relevant", "user_group"])
        for r in rows:
            writer.writerow([r["user_id"], r["strategy"], f"{r['ndcg_at_10']:.6f}", r["n_relevant"], r["user_group"]])


def _metadata_lines(meta: Dict) -> List[str]:
    m = meta
    return [
        f"- cut-off T: {iso(m['cutoff'])} (unix {m['cutoff']:.0f})",
        f"- seed: {m['seed']} ({N_BOOT} bootstrap resamples over users)",
        f"- users evaluated: {m['n_users']} (>= {MIN_LIKES_BEFORE} likes before T, >= 1 like after T on a snapshot post); "
        f"relevant likes: {m['n_relevant']}",
        f"- snapshot: {m['snapshot_posts']} posts, {m['snapshot_likes']} likes; removed at or after T: "
        f"{m['removed']['post']} posts, {m['removed']['like']} likes, {m['removed']['follow']} follows",
        f"- database fingerprint (sha256 of the rows): {m['fingerprint'][:16]}",
        f"- code: git commit {m['code']}",
        f"- relevant likes in the Week 5 lab's periods: train {m['after_in_lab_train']}, "
        f"validation {m['after_in_lab_val']}, test {m['after_in_lab_test']}",
    ]


def write_table(summary: Dict, meta: Dict, path: Path) -> None:
    lines = ["# Baseline table: NDCG@10 of the feed strategies", "",
             "Generated by `python scripts/evaluate.py`. Protocol: `docs/evaluation-protocol.md`.", "",
             "## Run", "", *_metadata_lines(meta), "",
             "## Mean NDCG@10 per strategy (95 % bootstrap interval over users)", "",
             "| strategy | mean NDCG@10 | 95 % interval | users |", "|---|---:|---:|---:|"]
    for s in STRATEGIES:
        r = summary["overall"][s]
        lines.append(f"| {s} | {r['diff']:.4f} | [{r['low']:.4f}, {r['high']:.4f}] | {r['n']} |")
    lines += ["", "## learned minus the best baseline (paired, 95 % interval)", "",
              "The best baseline is chosen in each slice: the simple feed with the highest mean there.", "",
              "| slice | users | best baseline | learned | baseline | difference | 95 % interval | verdict |",
              "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in summary["lifts"]:
        note = f" (n < {SMALL_SLICE})" if r["n"] < SMALL_SLICE else ""
        lines.append(f"| {r['slice']} | {r['n']} | {r['best_baseline']} | {r['learned']:.4f} | {r['baseline']:.4f} | "
                     f"{r['diff']:+.4f} | [{r['low']:+.4f}, {r['high']:+.4f}] | {r['verdict']}{note} |")
    lines += ["", "`wins`: the interval is above 0; `loses`: below 0; `unclear`: it contains 0.",
              "These numbers are not comparable with the Week 5 and Week 7 labs (one like hidden among 99 random "
              "items): here every strategy ranks the whole candidate set and every later like counts.", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", metadata={"Software": None})
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def write_dashboard(summary: Dict, meta: Dict, table_md: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    charts = []

    # 1. the four strategies with their intervals
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, s in enumerate(STRATEGIES):
        r = summary["overall"][s]
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["low"]], [r["high"] - r["diff"]]], fmt="o",
                    color=COLORS[s], capsize=5, ms=9)
    ax.set_yticks(range(len(STRATEGIES)), STRATEGIES)
    ax.set_xlim(0, max(0.05, max(summary["overall"][s]["high"] for s in STRATEGIES) * 1.25))
    ax.set_xlabel("mean NDCG@10 over users (95 % bootstrap interval)")
    ax.set_title("Which strategy puts the posts users will like into their top 10?")
    ax.grid(axis="x", alpha=0.3)
    charts.append(("Which strategy wins on average, and how sure are we?", _png(fig)))
    plt.close(fig)

    # 2. learned minus the best baseline, per slice, diverging colours centred on 0
    lifts = summary["lifts"]
    m = max(0.05, *(max(abs(r["low"]), abs(r["high"])) for r in lifts)) * 1.1
    norm, cmap = TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m), plt.get_cmap("RdBu")
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(lifts) + 1.6))
    for i, r in enumerate(lifts):
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["low"]], [r["high"] - r["diff"]]], fmt="o",
                    color=cmap(norm(r["diff"])), mec="#333333", ecolor="#444444", capsize=4, ms=9)
    ax.set_yticks(range(len(lifts)), [f"{r['slice']} vs {r['best_baseline']} (n={r['n']})" for r in lifts])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=1)
    ax.set_xlim(-m, m)
    ax.set_xlabel("NDCG@10: learned minus the best baseline of the slice (left of 0 = learned loses)")
    ax.set_title("Where does learned beat, or lose to, the best simple feed?")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="difference (diverging, centred on 0)")
    charts.append(("Where does learned lose? (learned minus the best baseline, per slice)", _png(fig)))
    plt.close(fig)

    # 3. every strategy per user-activity group, bars from 0 with intervals
    groups = [g for g in USER_GROUPS if g in summary["group_means"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(STRATEGIES)
    for j, s in enumerate(STRATEGIES):
        means = [summary["group_means"][g][s] for g in groups]
        x = np.arange(len(groups)) + (j - 1.5) * width
        ax.bar(x, [r["diff"] for r in means], width, color=COLORS[s], label=s,
               yerr=[[r["diff"] - r["low"] for r in means], [r["high"] - r["diff"] for r in means]], capsize=3)
    ax.set_xticks(range(len(groups)), [f"{g}\nn={summary['group_means'][g]['learned']['n']}" for g in groups])
    ax.set_ylim(0, None)
    ax.set_xlabel("user group (likes before the cut-off)")
    ax.set_ylabel("mean NDCG@10 (95 % interval)")
    ax.set_title("How does each strategy do for users we know little or much about?")
    ax.legend()
    charts.append(("Does the answer change with how much we know about the user?", _png(fig)))
    plt.close(fig)

    table_html = _markdown_tables_to_html(table_md)
    figures = "\n".join(f'<figure><figcaption>{html.escape(t)}</figcaption><img alt="{html.escape(t)}" '
                        f'src="data:image/png;base64,{b64}"></figure>' for t, b64 in charts)
    meta_html = "".join(f"<li>{html.escape(line[2:])}</li>" for line in _metadata_lines(meta))
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>InstaKill offline evaluation</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1d2433;line-height:1.45}}
h1{{margin-bottom:.2rem}} .sub{{color:#5a6478;margin-top:0}}
table{{border-collapse:collapse;margin:.8rem 0 1.6rem;font-size:.92rem}}
th,td{{border:1px solid #d5dae3;padding:.3rem .6rem}} th{{background:#f1f3f7;text-align:left}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
figure{{margin:1.6rem 0}} figcaption{{font-weight:600;margin-bottom:.4rem}} img{{max-width:100%}}
.loses{{color:#b2182b;font-weight:600}} .wins{{color:#2166ac;font-weight:600}}
</style></head><body>
<h1>InstaKill offline evaluation</h1>
<p class="sub">On our own data, which feed strategy puts the posts a user is going to like into their top 10, by how much, for which users, and how sure are we?</p>
<h2>Run</h2><ul>{meta_html}</ul>
{table_html}
<h2>Charts</h2>
{figures}
<p class="sub">Static page generated by <code>scripts/evaluate.py</code>; protocol and limitations in <code>docs/evaluation-protocol.md</code>.</p>
</body></html>
"""
    path.write_text(page, encoding="utf-8", newline="\n")


def _markdown_tables_to_html(md: str) -> str:
    """The two tables of baseline_table.md (with their headings) as HTML."""
    out, in_table = [], False
    for line in md.splitlines():
        if line.startswith("## ") and not line.startswith("## Run"):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-:"):
                continue
            tag = "th" if not in_table else "td"
            row = []
            for c in cells:
                cls = ' class="num"' if tag == "td" and (c[:1].isdigit() or c[:1] in "+-[") else ""
                text = html.escape(c)
                for word in ("loses", "wins"):
                    text = text.replace(word, f'<span class="{word}">{word}</span>')
                row.append(f"<{tag}{cls}>{text}</{tag}>")
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(row) + "</tr>")
        else:
            if in_table:
                out.append("</table>")
                in_table = False
    if in_table:
        out.append("</table>")
    return "\n".join(out)


# ---------------------------------------------------------------- command line

def evaluate(db: Path, cutoff: Optional[str], seed: int, out: Path, dashboard: bool = True, progress: bool = True) -> Dict:
    db = Path(db)
    if not db.exists():
        raise SystemExit(f"{db} not found")
    t = parse_cutoff(cutoff) if cutoff else default_cutoff(db)
    result = run_protocol(db, t, seed, progress=progress)
    summary = summarize(result["rows"], seed)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(result["rows"], out / "eval_results.csv")
    write_table(summary, result["meta"], out / "baseline_table.md")
    if dashboard:
        write_dashboard(summary, result["meta"], (out / "baseline_table.md").read_text(encoding="utf-8"),
                        out / "dashboard.html")
    return {"summary": summary, "meta": result["meta"]}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=ROOT / "instakill.db", help="database to evaluate (never modified)")
    parser.add_argument("--cutoff", help="ISO date/time of T, UTC by default (default: when 80%% of the likes had happened)")
    parser.add_argument("--seed", type=int, default=42, help="seed of the bootstrap")
    parser.add_argument("--out", type=Path, default=ROOT / "reports", help="folder for the three reports")
    args = parser.parse_args(argv)
    result = evaluate(args.db, args.cutoff, args.seed, args.out)
    print((args.out / "baseline_table.md").read_text(encoding="utf-8"))
    print(f"Wrote {args.out / 'eval_results.csv'}, {args.out / 'baseline_table.md'}, {args.out / 'dashboard.html'}")
    return result


if __name__ == "__main__":
    main()
