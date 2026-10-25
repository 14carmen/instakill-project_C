from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class SearchIndex:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.post_ids: List[int] = []
        self.embeddings: np.ndarray = np.array([])
        self.is_ready = False

    def build_index(self, posts: List[Dict]):
        if not posts:
            return
            
        self.post_ids = [post["id"] for post in posts]
        texts = [post["content"] for post in posts]
        
        self.embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        self.is_ready = True

    def encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode(query, convert_to_numpy=True)

search_index = SearchIndex()