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

def get_graph_defined_max_hops() -> int:
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)) as driver:
        with driver.session() as session:
            record = session.run("""
                MATCH (a), (b)
                WHERE elementId(a) <> elementId(b)
                WITH length(shortestPath((a)-[*]-(b))) AS hops
                RETURN max(hops) AS max_hops
            """).single()
            return record["max_hops"] or 3

def compute_hops(score: float) -> int:
    graph_max    = get_graph_defined_max_hops()
    effective    = min(graph_max, SYSTEM_MAX_HOPS)
    log_max      = math.log(effective + 1)
    scaled_score = score * log_max
    hops         = int(math.exp(scaled_score) - 1)
    return min(max(hops, 1), effective)