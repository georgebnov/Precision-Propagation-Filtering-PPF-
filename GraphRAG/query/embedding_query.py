import os
import sys
from typing import List
from openai import OpenAI
import cohere

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

def embedding_query(query: str) -> List[float]:
    try:
        response = client.embeddings.create(
            model = 'text-embedding-3-small',
            input = query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for query: {e}")
        return [0.0] * 1536