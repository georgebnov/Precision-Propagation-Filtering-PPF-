import os
import sys
from neo4j import GraphDatabase
from typing import List, Tuple
from openai import OpenAI
import math
import cohere
import joblib
from pathlib import Path

from Secret.secret import (
    NEO4J_URI_secret,
    NEO4J_USER_secret,
    NEO4J_PASS_secret,
    OPENAI_API_KEY_secret,
    COHERE_API_KEY_secret
)

NEO4J_URI = os.getenv("NEO4J_URI", NEO4J_URI_secret)
NEO4J_USER = os.getenv("NEO4J_USER", NEO4J_USER_secret)
NEO4J_PASS = os.getenv("NEO4J_PASS", NEO4J_PASS_secret)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY_secret)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", COHERE_API_KEY_secret)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
SIMILAR_K = int(os.getenv("SIMILAR_K", 3))
client = OpenAI(api_key=OPENAI_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY)
EMBEDDING_DIM = 1536
SYSTEM_MAX_HOPS = 10
TOP_K = 5 

def rerank_chunks_with_cohere(question: str, top_chunks: List[Tuple[str, str, float]], top_n: int = 5) -> List[Tuple[str, str, float]]:
    documents = [
    {"id": f"{i}", "text": "\n\n".join(chunk[1])}
    for i, chunk in enumerate(top_chunks)
]

    try:
        response = cohere_client.rerank(
            query=question,
            documents=documents,
            model="rerank-english-v3.0",
            top_n=top_n
        )
        reranked_chunks = []
        for result in response.results:
            idx = result.index 
            reranked_chunks.append((
                top_chunks[idx][0], 
                top_chunks[idx][1],
                result.relevance_score
            ))
            print("\nChunks passed to Cohere reranker:")
            for cid, _, sim in top_chunks:
                print(f"{cid} with similarity {sim:.4f}")
        return reranked_chunks
    except Exception as e:
        print(f"Error during Cohere rerank: {e}")
        return top_chunks[:top_n]