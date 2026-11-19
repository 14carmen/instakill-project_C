# Evaluation protocol (Week 7)

How `scripts/evaluate.py` measures the feed strategies of InstaKill on our own database. The numbers below come from the committed `reports/baseline_table.md` (MovieLens seed, default cut-off, seed 42).

## Question

On our own data, which feed strategy puts the posts a user is going to like into their top 10, by how much, for which users, and how sure are we? We compare four systems: `chronological`, `for_you` (the Week 4 formula) and `learned` (the Week 5 network), which are the strategies of `GET /feed`, and `popularity` (most likes in the last 365 days, ties by ascending post id), which exists only inside the evaluation script and is not served by the API.

## Cut-off and snapshot

We pick a cut-off time `T`. The default is the moment by which 80 % of the likes had happened: 2016-02-04 21:34:53 UTC on the MovieLens seed. The script copies the database to a temporary file (the working `instakill.db` is never modified) and deletes from the copy every post, like and follow created at or after `T`, plus any like whose post was deleted. On the MovieLens seed this removes 1 908 posts and 9 716 likes and leaves 7 816 posts and 38 864 likes. Our seeder creates no follows, so no follow rule can see the future; follows created through the API are cut at `T` like everything else. Users are kept: their sign-up time is the time the seeder ran, so it says nothing about the past.

## Users

We evaluate every user with at least 3 likes before `T` and at least 1 like after `T` on a post that exists in the snapshot (a post created before `T`). On the MovieLens seed that is only **25 users**: most MovieLens users rated their movies in one or two sessions and never came back, so few of them have both a past and a future around 2016.

## Candidates and the question we ask

For each user and strategy we call `app.service.build_user_feed`, the same function `GET /feed` calls, on the snapshot with `now = T`, and keep the top 10. The candidates are therefore exactly the product's: posts not written by the user and not already liked by them (restricted to followed authors when the user follows someone). The script does not re-implement any strategy; `popularity` re-orders the candidate list that the service returns.

## Relevance

The relevant posts of a user are the snapshot posts they liked after `T` (469 likes in total, 1 to 71 per user). A post the user liked after `T` but that was created after `T` is not in the snapshot, so it can be neither recommended nor counted.

## Metric

NDCG@10 with several relevant posts: `DCG = Σ rel_i / log2(i + 2)` over the 10 returned positions (`rel_i` = 1 for a relevant post), divided by the DCG of a perfect top 10 that puts `min(#relevant, 10)` relevant posts first. One number per user, between 0 and 1, then averaged over users so that every user counts once, however many likes they gave.

These numbers are **not comparable** with the Week 5 and Week 7 labs. There, each moment hides one like among 99 random items. Here every strategy ranks the whole candidate set (several thousand posts) and every later like counts.

## Intervals

A 95 % paired bootstrap over **users** with a fixed seed (42, 2 000 resamples): we resample users with replacement and keep the same users for both strategies of a comparison. We report an interval for each strategy's mean and for `learned` minus the best baseline. The verdict is `wins` when the interval is above 0, `loses` when it is below 0 and `unclear` when it contains 0. We never write *better* without an interval that excludes 0.

## Slices

1. **User activity**, counted before `T`: `new (<10 likes)`, `casual (10-49)`, `regular (50-199)`, `power (200+)`, as in the lab.
2. **Taste for new posts**: whether at least half of the user's relevant posts were less than a year old at `T` (`mostly new posts (<1 year)`) or not (`mostly older posts`). It asks the question of limitation 2 directly.

In every slice, `learned` is compared with the best of the three simple feeds *in that slice*. With 25 users, every slice has fewer than 20 users, so its interval matters more than its verdict.

## Known limitations

1. **The model may have seen the future.** `like_model.npz` was trained in the Week 5 lab on MovieLens likes before 2013-03-19 (the lab's `val` cut-off). Its hyper-parameters were chosen on the lab's validation period (2013-03-19 to 2016-07-10). None of our 469 evaluation likes fall in the training period. However, 82 of them (17 %) fall in the validation period that we tuned on, and 387 (83 %) in the lab's test period, which nothing was tuned on. `learned`'s score can therefore be slightly optimistic. A cleaner run would put the cut-off after 2016-07-10, at the price of even fewer users.
2. **The population is not InstaKill's.** Here, only 63 of the 469 relevant likes (13 %) go to posts less than a year old, and the median liked post is 9 years old at `T`. On a social network most likes go to posts from the last few days. Only 2 of our 25 users like mostly new posts, so our own slice cannot answer the question. The lab's Failure Atlas can (with our Week 4 weights 2.0 / 1.5 / 1.0 for `for_you`): on movies less than a year old, `learned` lost to `for_you` by 0.400 NDCG@10 on validation (95 % interval −0.469 to −0.328, 104 moments) and by 0.282 on test (−0.360 to −0.202, 89 moments), the only group out of 8 where it loses, because it trusts a long history of likes that a new post has not had time to collect. On a feed of fresh posts we would therefore expect `for_you` (freshness plus affinity), not `learned`, to win. Before serving `learned` to everybody, we would route new posts to `for_you` or retrain on fresh data.
3. **Few users, wide intervals.** With 25 users the interval of `learned` minus `popularity` overall spans about ±0.05. A difference smaller than that cannot be told from noise on this database.

## Reproducing the table

```bash
pip install -e ".[dev,eval]"
python scripts/seed_movielens.py ml-latest-small.zip     # same rows -> same fingerprint
python scripts/evaluate.py                               # writes reports/
```

`eval_results.csv` and `baseline_table.md` are byte-identical for the same database and arguments. The run metadata records the database fingerprint (a sha256 of the rows, not of the file) and the last commit that changed `app/`, `scripts/` or `pyproject.toml`. Committing the reports therefore does not change it, and a fresh clone reproduces the committed table exactly.
