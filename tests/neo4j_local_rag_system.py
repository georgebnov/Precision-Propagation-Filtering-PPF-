import os
import sys
from pathlib import Path
from typing import List

from GraphRAG.Secret.secret import (
    NEO4J_URI_secret,
    NEO4J_USER_secret,
    NEO4J_PASS_secret,
    OPENAI_API_KEY_secret,
    COHERE_API_KEY_secret
)

# === Dependency check for PDF reader ===
try:
    from PyPDF2 import PdfReader  # common install: pip install PyPDF2
except ModuleNotFoundError:
    try:
        from pypdf import PdfReader  # alternative library: pip install pypdf
    except ModuleNotFoundError:
        print("Error: Missing PDF library. Install with `pip install PyPDF2` or `pip install pypdf`.")
        sys.exit(1)

# Neo4j driver and OpenAI client
from neo4j import GraphDatabase
import openai



# === CONFIGURATION ===
NEO4J_URI = os.getenv("NEO4J_URI", NEO4J_URI_secret)
NEO4J_USER = os.getenv("NEO4J_USER", NEO4J_USER_secret)
NEO4J_PASS = os.getenv("NEO4J_PASS", NEO4J_PASS_secret)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY_secret)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", COHERE_API_KEY_secret)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))  # words per chunk

# Validate OpenAI API key\
if not OPENAI_API_KEY:
    print("Error: Missing OPENAI_API_KEY environment variable.")
    sys.exit(1)

# === INITIALIZE CLIENTS ===
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
except Exception as e:
    print(f"Error connecting to Neo4j at {NEO4J_URI}: {e}")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

# === HELPER FUNCTIONS ===

def load_text_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split raw text into chunks of up to chunk_size words.
    """
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def load_pdf_chunks(path: Path) -> List[str]:
    """
    Extract text from a PDF and return word chunks.
    """
    reader = PdfReader(str(path))
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return load_text_chunks(text)


def load_txt_chunks(path: Path) -> List[str]:
    """
    Read a text or markdown file and return word chunks.
    """
    text = path.read_text(encoding="utf-8")
    return load_text_chunks(text)


def ingest_file(file_path: str):
    """
    Chunk and ingest a local file into Neo4j.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        chunks = load_pdf_chunks(path)
    elif suffix in [".txt", ".md"]:
        chunks = load_txt_chunks(path)
    else:
        print(f"Unsupported file type: {suffix}")
        return

    with driver.session() as session:
        for idx, chunk in enumerate(chunks, start=1):
            node_id = f"{path.stem}_{idx}"
            session.run(
                "MERGE (c:Chunk {id: $id}) SET c.content = $content, c.source = $source",
                id=node_id, content=chunk, source=path.name
            )
    print(f"Ingested {len(chunks)} chunks from {path.name}")


def retrieve_chunks(question: str, limit: int = 3) -> List[str]:
    """
    Return up to `limit` chunks containing any keyword from the question.
    """
    keywords = [w.lower() for w in question.split() if len(w) > 2]
    if not keywords:
        return []

    cypher = (
        "MATCH (c:Chunk) " 
        "WHERE any(kw IN $keywords WHERE toLower(c.content) CONTAINS kw) "
        "RETURN c.content AS content LIMIT $limit"
    )
    with driver.session() as session:
        result = session.run(cypher, keywords=keywords, limit=limit)
        return [rec["content"] for rec in result]


def ask_openai(question: str, chunks: List[str]) -> str:
    """
    Send prompt with context chunks to OpenAI and return the answer.
    """
    context = "\n\n".join(chunks)
    prompt = (
        "Answer the question using ONLY the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI request failed: {e}"

# === CLI ===

def main():
    if len(sys.argv) < 3:
        print("Usage: script.py [ingest|ask] <file_path|question>")
        return
    cmd, arg = sys.argv[1], sys.argv[2]

    if cmd == "ingest":
        ingest_file(arg)
    elif cmd == "ask":
        chunks = retrieve_chunks(arg)
        if not chunks:
            print("No relevant chunks found.")
        else:
            print("Retrieved chunks, querying LLM...")
            print(ask_openai(arg, chunks))
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()

# === Requirements ===
# pip install PyPDF2 neo4j openai
# or pip install pypdf4 neo4j openai
