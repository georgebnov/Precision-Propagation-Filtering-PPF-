import os
import sys
from neo4j import GraphDatabase
from typing import List, Tuple
from openai import OpenAI
import math
import cohere
import joblib
from pathlib import Path
import embedding_query
from compute_max_hops import compute_hops
import cosine_similarity
from top_k import get_top_k_paths_precise
from rerank_cohere import rerank_chunks_with_cohere
from generate_answer import generate_answer_from_chunks
from top_k import get_top_k_paths_precise
from precision_expander import process_paths_for_ppf, filter_by_precision, build_llm_context_from_chains

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from GraphRag_retrieval_controller.predict_controller import (
    embedding_query,
    predict_broadness_score
)

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

question = input("Enter your question: ")
query_emb = embedding_query(question)
score = predict_broadness_score(query_emb)
print(f"Broadness score: {score:.2f}")
hops = compute_hops(score)
print(f"Using {hops} hops for retrieval.")

retrieved_paths = get_top_k_paths_precise(query_emb, k=TOP_K, hops=hops)
structured_chains = process_paths_for_ppf(query_emb, retrieved_paths)

filtered_chains = filter_by_precision(structured_chains, threshold=0.8, top_n=TOP_K)

reranked = rerank_chunks_with_cohere(question, filtered_chains)
answer = generate_answer_from_chunks(question, reranked)

print("\n=== Answer ===")
print(answer)