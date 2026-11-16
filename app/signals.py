"""The six signals of the learned feed, exactly as `LiveSignals.features` computes them in week05.ipynb.

The network in like_model.npz only understands inputs built with these formulas and constants.
Any difference (another half-life, a log where the notebook has a ratio, the Week 4 affinity)
is train/serve skew: the model still returns numbers, but they no longer mean anything.

| signal     | formula (notebook)                                                       |
|------------|--------------------------------------------------------------------------|
| freshness  | 0.5 ** (age_days / half_life_days)                                       |
| affinity   | (post_genres . user_genre_counts) / (user_genre_total * n_genres(post))  |
| popularity | likes in the last popularity_days / max of that over all posts (>= 1)    |
| activity   | min(1, log1p(user's likes) / log1p(activity_cap))                        |
| lifetime   | log1p(likes ever) / log1p(max likes ever over all posts, >= 1)           |
| age        | min(1, age_days / age_cap_days)                                          |

Only likes strictly before `now` count, as in the notebook. Everything is vectorised: no Python
loop over posts.
"""
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

DAY = 86_400
FEATURES = ("freshness", "affinity", "popularity", "activity", "lifetime", "age")

# The 19 MovieLens genres the notebook builds its genre matrix from ("(no genres listed)" excluded).
GENRES = (
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
_GENRE_INDEX = {name: j for j, name in enumerate(GENRES)}


@dataclass(frozen=True)
class SignalConfig:
    """The constants behind the formulas. They travel inside like_model.npz with the weights."""
    half_life_days: float = 3 * 365
    popularity_days: float = 365
    activity_cap: float = 500
    age_cap_days: float = 25 * 365

    @classmethod
    def from_saved(cls, saved: Mapping[str, np.ndarray]) -> "SignalConfig":
        return cls(**{name: float(saved[name]) for name in cls.__dataclass_fields__})


def genre_matrix(genre_strings: Iterable[str]) -> np.ndarray:
    """["Comedy|Drama", "", ...] -> (n, 19) array with 1.0 in the columns of each post's genres."""
    genre_strings = list(genre_strings)
    matrix = np.zeros((len(genre_strings), len(GENRES)))
    for i, text in enumerate(genre_strings):
        for name in (text or "").split("|"):
            j = _GENRE_INDEX.get(name.strip())
            if j is not None:
                matrix[i, j] = 1.0
    return matrix


def compute_features(
    *,
    now: float,
    created_ts: np.ndarray,          # (P,) creation time of every post
    genres: np.ndarray,              # (P, 19) genre matrix of every post
    like_post_idx: np.ndarray,       # (L,) index into the P posts of every like before `now`
    like_ts: np.ndarray,             # (L,) time of every like before `now`
    user_like_mask: np.ndarray,      # (L,) True for the likes of the user the feed is for
    candidate_idx: np.ndarray,       # (C,) indices of the posts to score
    config: SignalConfig,
) -> np.ndarray:
    """(C, 6) signals for the candidates, columns in the order of FEATURES."""
    n_posts = len(created_ts)
    lifetime_counts = np.bincount(like_post_idx, minlength=n_posts).astype(float)
    recent_mask = like_ts >= now - config.popularity_days * DAY
    recent_counts = np.bincount(like_post_idx[recent_mask], minlength=n_posts).astype(float)

    user_genre_counts = genres[like_post_idx[user_like_mask]].sum(axis=0)        # (19,)
    user_n_likes = int(user_like_mask.sum())

    rows = np.asarray(candidate_idx, dtype=int)
    age_days = (now - created_ts[rows]) / DAY
    freshness = 0.5 ** (age_days / config.half_life_days)

    total = user_genre_counts.sum()
    if total > 0:
        n_genres = np.maximum(genres[rows].sum(axis=1), 1.0)
        affinity = (genres[rows] @ user_genre_counts) / (total * n_genres)
    else:
        affinity = np.zeros(len(rows))                                              # nothing known about this user

    popularity = recent_counts[rows] / max(recent_counts.max(initial=0.0), 1.0)
    activity = np.full(len(rows), min(1.0, np.log1p(user_n_likes) / np.log1p(config.activity_cap)))
    lifetime = np.log1p(lifetime_counts[rows]) / np.log1p(max(lifetime_counts.max(initial=0.0), 1.0))
    age = np.minimum(1.0, age_days / config.age_cap_days)
    return np.column_stack([freshness, affinity, popularity, activity, lifetime, age])
