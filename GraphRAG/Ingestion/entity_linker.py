#!/usr/bin/env python3
# ---------------------------------------
# entity_linker.py
# ---------------------------------------
# Extracts entities from ingested chunks,
# links entities to chunks, and creates SIMILAR_TO edges
# for a GraphRAG pipeline in Neo4j (OpenAI v1.x syntax)
# ---------------------------------------

import os
import sys
import json
import re
import math
from typing import List, Tuple
from neo4j import GraphDatabase
from openai import OpenAI
import argparse

from Secret.secret import (
    NEO4J_URI_secret,
    NEO4J_USER_secret,
    NEO4J_PASS_secret,
    OPENAI_API_KEY_secret,
    COHERE_API_KEY_secret
)

from openai import OpenAI
import cohere

client = OpenAI(api_key=OPENAI_API_KEY_secret)
cohere_client = cohere.Client(COHERE_API_KEY_secret)


# === CONFIGURATION ===

NEO4J_URI = os.getenv("NEO4J_URI", NEO4J_URI_secret)
NEO4J_USER = os.getenv("NEO4J_USER", NEO4J_USER_secret)
NEO4J_PASS = os.getenv("NEO4J_PASS", NEO4J_PASS_secret)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY_secret)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", COHERE_API_KEY_secret)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
SIMILAR_K = int(os.getenv("SIMILAR_K", 7))

# Validate OpenAI API key
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY must be set as an environment variable or in this file.")
    sys.exit(1)

# Initialize clients
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
client = OpenAI(api_key=OPENAI_API_KEY)

# === ENTITY EXTRACTION ===
def extract_entities(text: str) -> List[str]:
    prompt = (
        "Extract a JSON array of unique entities (e.g., benefits, coverages, plan names, important terms) "
        "from the following text. Return ONLY the JSON array with no explanation.\n\n"
        f"Text:\n{text}\n\nEntities:"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"(\[.*\])", raw, re.DOTALL)
        if match:
            entities = json.loads(match.group(1))
            if isinstance(entities, list):
                return [e.strip() for e in entities if isinstance(e, str) and e.strip()]
    except Exception as e:
        print(f"Warning: entity extraction failed: {e}")
    return []

# === SIMILARITY UTILITIES ===
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

# === PROCESS AND LINK CHUNKS + ENTITIES ===
def process_chunks_and_entities():
    with driver.session() as session:
        result = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.content AS content")
        for record in result:
            chunk_id = record["id"]
            content = record["content"]
            print(f"Processing chunk: {chunk_id}")
            entities = extract_entities(content)
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    WITH e
                    MATCH (c:Chunk {id: $id})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    """,
                    name=entity,
                    id=chunk_id
                )
            print(f"Linked entities: {entities}\n")
    print("\n✅ Entity extraction and linking complete.")

# === PROCESS AND LINK SIMILARITY ===
def process_similarity():
    SIMILARITY_THRESHOLD = 0.75 
    TOP_K = SIMILAR_K  

    with driver.session() as session:
        rows = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.embedding AS emb")
        data: List[Tuple[str, List[float]]] = []
        for r in rows:
            data.append((r["id"], r["emb"]))

    for id_a, emb_a in data:
        sims = []
        for id_b, emb_b in data:
            if id_a == id_b:
                continue
            score = cosine_similarity(emb_a, emb_b)
            sims.append((id_b, score))

        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)

        links_added = 0
        with driver.session() as sess:
            for id_b, score in sims_sorted:
                if score < SIMILARITY_THRESHOLD:
                    # Skip low similarity links
                    continue
                if links_added >= TOP_K:
                    # Reached our desired number of links
                    break
                sess.run(
                    """
                    MATCH (a:Chunk {id: $id_a}), (b:Chunk {id: $id_b})
                    MERGE (a)-[r:SIMILAR_TO]->(b)
                    SET r.score = $score
                    """,
                    id_a=id_a,
                    id_b=id_b,
                    score=score
                )
                links_added += 1
        print(f"Linked {links_added} similar chunks for {id_a} (threshold: {SIMILARITY_THRESHOLD})")

    print("\nSimilarity linking complete.")


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity extraction and similarity linking for GraphRAG.")
    parser.add_argument("--entities", action="store_true", help="Extract and link entities to chunks.")
    parser.add_argument("--similar", action="store_true", help="Calculate and link top-K similar chunks.")
    args = parser.parse_args()

    if not args.entities and not args.similar:
        parser.print_help()
        sys.exit(0)

    if args.entities:
        process_chunks_and_entities()
    if args.similar:
        process_similarity()
