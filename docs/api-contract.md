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