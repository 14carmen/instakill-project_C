import time
from contextlib import asynccontextmanager
from typing import Literal, List, Optional, Any

from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel

from app.db import repository
from app.db.database import init_db
from app.search_index import search_index
from app.service import ModelUnavailableError, UserNotFoundError, build_user_feed, search_posts_by_meaning

class UserCreate(BaseModel):
    username: str

class UserUpdate(BaseModel):
    username: Optional[str] = None

class PostCreate(BaseModel):
    user_id: int
    content: str

class PostUpdate(BaseModel):
    content: Optional[str] = None

class FollowAction(BaseModel):
    follower_id: int
    followed_id: int

class LikeAction(BaseModel):
    user_id: int
    post_id: int
    created_at: Optional[float] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    posts = repository.get_all_posts()
    if posts:
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
    return {"query": q.strip(), "results": results}

@app.get("/feed")
def get_feed(
    user_id: int = Query(..., description="User ID requesting the feed"),
    limit: int = Query(20, ge=1, le=50, description="Max posts to return (1..50)"),
    strategy: Literal["for_you", "chronological", "learned"] = Query(
        "for_you", description="Ranking strategy: for_you, chronological, or learned"
    ),
    as_of: Optional[float] = Query(
        None, ge=0, description="Optional unix time: replay the feed as it was then (default: now)"
    ),
):
    try:
        return build_user_feed(user_id=user_id, limit=limit, strategy=strategy, now=as_of)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc



# USERS
@app.post("/users", status_code=201)
def create_user(user: UserCreate):
    new_user = repository.create_user(username=user.username)
    return {"message": "User created", "data": new_user}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    user = repository.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users")
def list_users():
    return repository.get_all_users()

@app.patch("/users/{user_id}")
def update_user(user_id: int, user: UserUpdate):
    updated = repository.update_user(user_id, username=user.username)
    return {"message": "User updated", "data": updated}

@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    repository.delete_user(user_id)
    return {"message": "User deleted"}


# POSTS
@app.post("/posts", status_code=201)
def create_post(post: PostCreate):
    new_post = repository.create_post(user_id=post.user_id, content=post.content)
    return {"message": "Post created", "data": new_post}

@app.get("/posts/{post_id}")
def get_post(post_id: int):
    post = repository.get_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

@app.get("/posts")
def list_posts():
    return repository.get_all_posts()

@app.patch("/posts/{post_id}")
def update_post(post_id: int, post: PostUpdate):
    updated = repository.update_post(post_id, content=post.content)
    return {"message": "Post updated", "data": updated}

@app.delete("/posts/{post_id}")
def delete_post(post_id: int):
    repository.delete_post(post_id)
    return {"message": "Post deleted"}


# FOLLOWS
@app.post("/follows", status_code=201)
def follow_user(action: FollowAction):
    repository.create_follow(follower_id=action.follower_id, followed_id=action.followed_id)
    return {"message": "Followed successfully"}

@app.delete("/follows")
def unfollow_user(action: FollowAction):
    repository.delete_follow(follower_id=action.follower_id, followed_id=action.followed_id)
    return {"message": "Unfollowed successfully"}

@app.get("/users/{user_id}/following")
def list_following(user_id: int):
    following = repository.get_following(user_id)
    return {"user_id": user_id, "following": following}

@app.get("/users/{user_id}/followers")
def list_followers(user_id: int):
    followers = repository.get_followers(user_id)
    return {"user_id": user_id, "followers": followers}


# LIKES
@app.post("/likes", status_code=201)
def like_post(action: LikeAction):
    like_time = action.created_at if action.created_at else time.time()
    repository.create_like(user_id=action.user_id, post_id=action.post_id, created_at=like_time)
    return {"message": "Post liked"}

@app.delete("/likes")
def unlike_post(action: LikeAction):
    repository.delete_like(user_id=action.user_id, post_id=action.post_id)
    return {"message": "Post unliked"}