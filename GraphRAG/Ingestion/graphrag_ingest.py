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
from beir.datasets.data_loader import GenericDataLoader

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

print("🧼 Temporary files deleted.")

def load_text_chunks(text: str, chunk_size: int = 380, overlap: int = 80) -> List[str]:

    words = text.split()
    stride = chunk_size - overlap
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), stride)]
    return chunks

def load_txt_chunks(path:Path) -> List[str]: 

    with open(path, 'r', encoding='utf=8') as f:
        text = f.read()
        return load_text_chunks(text)
    
def load_pdf_chunks(path:Path) -> List[str]: 

    reader = PdfReader(str(path)) 
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return load_text_chunks(text)

def load_chunks_from_file(file_path:str) -> List[str]: 

    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} does not exist.")
        return []

    if path.suffix.lower() == '.pdf':
        return load_pdf_chunks(path)
    elif path.suffix.lower() in ['.txt', '.md']:
        return load_txt_chunks(path)
    else:
        print(f"Error: Unsupported file type {path.suffix}. Only .txt, .md, and .pdf are supported.")
        return []
    
def generate_embeddings_for_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    for chunk in tqdm(chunks, desc="Generating embeddings", unit="chunk"):
        try:
            response = client.embeddings.create(
                model='text-embedding-3-small',
                input= chunk
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
            embeddings.append([0.0] * 1536)
    return embeddings

def ingest_chunks_to_neo4j(chunks: List[str], embeddings: List[List[float]], file_name: str):
    with driver.session() as session:
        print(f"Chunks: {len(chunks)}") 
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{file_name}_chunk_{i}"
            session.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.content = $content, c.embedding = $embedding
                """,
                id=chunk_id,
                content=chunk,
                embedding=embedding
            )
            print(f"Ingested {chunk_id}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python graphrag_ingest.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_name = Path(file_path).stem

    chunks = list(load_chunks_from_file(file_path))
    embeddings = generate_embeddings_for_chunks(chunks)
    ingest_chunks_to_neo4j(chunks, embeddings, file_name)