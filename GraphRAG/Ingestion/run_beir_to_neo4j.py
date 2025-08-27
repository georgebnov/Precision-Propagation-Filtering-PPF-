import os
import sys
from pathlib import Path
from typing import List
from neo4j import GraphDatabase
from openai import OpenAI
import numpy as np
from PyPDF2 import PdfReader
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Secret import secret
from beir.beir.datasets.data_loader import GenericDataLoader
from graphrag_ingest import (
    load_chunks_from_file,
    generate_embeddings_for_chunks,
    ingest_chunks_to_neo4j
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
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
client = OpenAI(api_key=OPENAI_API_KEY)

# Step 1: Load BEIR dataset (Scifact is small and good for first run)
corpus, queries, qrels = GenericDataLoader("../datasets/scifact", corpus_file="corpus.jsonl", query_file="queries.jsonl", qrels_folder="qrels").load(split="test")

# Step 2: Create a temporary folder for fake document files
temp_dir = Path("beir_temp_docs")
temp_dir.mkdir(exist_ok=True)

# Step 3: Loop through BEIR documents and simulate file ingestion
for doc_id, doc in corpus.items():
    file_path = temp_dir / f"{doc_id}.txt"
    file_name = doc_id  # used as Neo4j node ID prefix

    # Save title + text to a temporary file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{doc['title']}\n{doc['text']}")

    # Use your ingestion pipeline
    chunks = load_chunks_from_file(str(file_path))  # -> List[str]
    embeddings = generate_embeddings_for_chunks(chunks)  # -> List[List[float]]
    ingest_chunks_to_neo4j(chunks, embeddings, file_name)

print("✅ All BEIR documents ingested into Neo4j.")

# Step 4: Clean up temporary files
import shutil
shutil.rmtree(temp_dir)
print("🧼 Temporary files deleted.")