from typing import List, Dict
from app.search_index import search_index
from app.retrieval import get_top_k_results

def search_posts_by_meaning(query: str, limit: int, all_posts_db: List[Dict]) -> List[Dict]:
    if not search_index.is_ready or not all_posts_db:
        return []
        
    query_emb = search_index.encode_query(query)
    top_results = get_top_k_results(query_emb, search_index.embeddings, search_index.post_ids, limit)
    
    posts_by_id = {post["id"]: post for post in all_posts_db}
    
    final_results = []
    for post_id, score in top_results:
        post_data = posts_by_id.get(post_id)
        if post_data:
            final_results.append({
                "post_id": post_id,
                "user_id": post_data["user_id"],
                "content": post_data["content"],
                "score": round(score, 4)
            })
            
    return final_results