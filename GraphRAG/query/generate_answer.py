import os
from typing import List, Tuple
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

def generate_answer_from_chunks(question: str,  reranked_chunks: List[Tuple[str, str, float]]) -> str:
    context = "\n\n".join(
    "\n\n".join(chunk[1]) if isinstance(chunk[1], list) else chunk[1]
    for chunk in reranked_chunks
)
    prompt = (
        "You are a precise assistant helping users with questions about their insurance.\n"
        "You must ONLY use the provided context below to answer the question.\n"
        "Do NOT mention checking websites, calling, or disclaimers unless explicitly found in the context.\n"
        "Provide clear, actionable information extracted directly from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
)

    print("Prompt prepared for the model.\n")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer at this time."