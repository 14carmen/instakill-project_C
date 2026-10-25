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