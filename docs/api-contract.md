# API Contract: Semantic Search

This contract describes the endpoint that will be implemented for searching posts.

## Request
`GET /search`

**Query Parameters:**
- `q` (string, required): The search query text. Cannot be empty or whitespace only.
- `limit` (integer, required): The maximum number of results to return. Must be between 1 and 20.

## Successful Response (200 OK)
The response returns the original query and a list of results ordered by descending similarity score.

```json
{
  "query": "cheap dinner",
  "results": [
    {
      "post_id": 105,
      "user_id": 4,
      "content": "Found a great budget-friendly restaurant today!",
      "score": 0.89
    },
    {
      "post_id": 12,
      "user_id": 2,
      "content": "Eating fast food to save money.",
      "score": 0.74
    }
  ]
}
```

## GET /feed

Returns a ranked feed of posts for a given user.

### Query Parameters
- `user_id` (integer, required): ID of the user requesting the feed.
- `limit` (integer, optional, default `20`, range `1..50`): Maximum number of posts to return.
- `strategy` (string, optional, default `"for_you"`):
  - `"chronological"`: newest first, no score;
  - `"for_you"`: the Week 4 hand-tuned formula `2.0·freshness + 1.5·affinity + 1.0·popularity`;
  - `"learned"`: the Week 5 neural network (`app/like_model.npz`); `score` is the predicted probability of a like, between 0 and 1.
- `as_of` (number, optional, unix seconds, `>= 0`): replay the feed as it was at that moment: only posts created and likes given before `as_of` are seen. Default: the current time. Useful with MovieLens data, whose likes end in 2018.

Candidates are the same for every strategy: posts not written by the user and not already liked by them, restricted to followed authors when the user follows somebody who has eligible posts (cold start: all eligible posts). Ties are broken by ascending `post_id`.

### Responses
- **200 OK**:
  ```json
  {
    "user_id": 1,
    "strategy": "learned",
    "results": [
      {
        "post_id": 7,
        "user_id": 3,
        "content": "heapq makes top-k selection fast and memory efficient",
        "created_at": "2026-10-25 17:00:00",
        "score": 0.7342
      }
    ]
  }
  ```
- **404 Not Found**: the user does not exist.
- **422 Unprocessable Entity**: `limit` out of range, unknown `strategy`, or negative `as_of`.
- **503 Service Unavailable**: `strategy=learned` but `like_model.npz` could not be loaded.
