import numpy as np
from typing import List

def calculate_cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    
    dot_products = np.dot(doc_embs, query_emb)
    query_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(doc_embs, axis=1)
    
    denominators = query_norm * doc_norms
    denominators[denominators == 0] = 1e-10
    
    return dot_products / denominators

def get_top_k_results(query_emb: np.ndarray, doc_embs: np.ndarray, post_ids: List[int], k: int) -> List[tuple]:
    
    if len(post_ids) == 0:
        return []

    scores = calculate_cosine_similarity(query_emb, doc_embs)
    
    dtype = [('neg_score', float), ('id', int)]
    sort_data = np.array(list(zip(-scores, post_ids)), dtype=dtype)
    
    sorted_indices = np.argsort(sort_data)
    top_indices = sorted_indices[:k]
    
    return [(post_ids[i], float(scores[i])) for i in top_indices]