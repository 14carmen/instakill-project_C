
"""
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from app.search_index import search_index
from app.service import search_posts_by_meaning
import sqlite3

def get_all_posts_from_db():
    conn = sqlite3.connect("instakill.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_id, content FROM post")
    posts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return posts

@asynccontextmanager
async def lifespan(app: FastAPI):
    posts = get_all_posts_from_db()
    search_index.build_index(posts)
    yield

app = FastAPI(lifespan=lifespan)
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/search")
def semantic_search(
    q: str = Query(..., min_length=1, description="Non-empty search query"),
    limit: int = Query(10, ge=1, le=20, description="Results limit between 1 and 20")
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    posts_db = get_all_posts_from_db()
    results = search_posts_by_meaning(q.strip(), limit, posts_db)
    
    return {
        "query": q.strip(),
        "results": results
    }

"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from app.db import repository
from app.db.database import init_db
from app.search_index import search_index
from app.service import UserNotFoundError, build_user_feed, search_posts_by_meaning


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    posts = repository.get_all_posts()
    search_index.build_index(posts)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/search")
def semantic_search(
    q: str = Query(..., min_length=1, description="Non-empty search query"),
    limit: int = Query(10, ge=1, le=20, description="Results limit between 1 and 20"),
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    posts_db = repository.get_all_posts()
    results = search_posts_by_meaning(q.strip(), limit, posts_db)

    return {
        "query": q.strip(),
        "results": results,
    }


@app.get("/feed")
def get_feed(
    user_id: int = Query(..., description="User ID requesting the feed"),
    limit: int = Query(20, ge=1, le=50, description="Max posts to return (1..50)"),
    strategy: Literal["for_you", "chronological"] = Query(
        "for_you", description="Ranking strategy: for_you or chronological"
    ),
):
    try:
        return build_user_feed(user_id=user_id, limit=limit, strategy=strategy)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc