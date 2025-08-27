import os
import sys
from neo4j import GraphDatabase
from typing import List, Tuple
from openai import OpenAI
import math
import cohere
import joblib
from pathlib import Path
import numpy as np
from cosine_similarity import cosine_similarity

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

def get_top_k_paths_precise(
    query_embedding: List[float],
    k: int,
    hops: int
) -> List[Tuple[List[str], List[str]]]:
    """
    Retrieve top-k anchors by vector similarity, then for each anchor,
    find all possible paths with exactly `hops` expansions,
    returning structured paths (ids_chain, contents_chain) for PPF enrichment.
    """
    paths = []

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)) as driver:
        with driver.session() as session:
            # Retrieve top-k anchors
            result = session.run(
                """
                CALL db.index.vector.queryNodes('chunk_embedding_index', $k, $query_embedding)
                YIELD node, score
                RETURN node, score
                ORDER BY score DESC
                LIMIT $k
                """,
                {"k": k, "query_embedding": query_embedding}
            )

            for record in result:
                anchor_node = record["node"]
                anchor_id = anchor_node["id"]
                anchor_content = anchor_node["content"]

                # Retrieve all paths from anchor with exactly `hops` hops (paths of length hops + 1)
                path_result = session.run(
                    """
                    MATCH p = (anchor:Chunk)-[:SIMILAR_TO*{min_hops}..{max_hops}]->(end:Chunk)
                    WHERE elementId(anchor) = $node_id
                    WITH nodes(p) AS path_nodes
                    RETURN [n IN path_nodes | n.id] AS ids,
                        [n IN path_nodes | n.content] AS contents,
                        [n IN path_nodes | n.embedding] AS embeddings
                    """.format(min_hops=hops, max_hops=hops),
                    {"node_id": anchor_node.element_id}
                )

                for path_record in path_result:
                    ids_chain = path_record["ids"]
                    contents_chain = path_record["contents"]
                    embeddings_chain = path_record["embeddings"]
                    anchor_embedding = anchor_node["embedding"]

                    # Ensure the anchor is included explicitly if needed
                    if ids_chain[0] != anchor_id:
                        ids_chain = [anchor_id] + ids_chain
                        contents_chain = [anchor_content] + contents_chain
                        embeddings_chain = [anchor_embedding] + embeddings_chain

                    paths.append((ids_chain, contents_chain, embeddings_chain))


    print(f"\n✅ Retrieved {len(paths)} total paths across {k} anchors using {hops} hops.")
    return paths