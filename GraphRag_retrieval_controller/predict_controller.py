# GraphRAG_retreival_controller/predict_controller.py

import numpy as np
import joblib
from openai import OpenAI
from typing import List
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
pca_path = BASE_DIR / "Models" / "pca.joblib"
regressor_path = BASE_DIR / "Models" / "regression.joblib"

print(f"Loading PCA from: {pca_path}")
print(f"Loading regressor from: {regressor_path}")

pca = joblib.load(pca_path)
regressor = joblib.load(regressor_path)
client    = OpenAI()

def embedding_query(query: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

def predict_broadness_score(embedding: List[float]) -> float:
    arr         = np.array(embedding).reshape(1, -1)
    emb_reduced = pca.transform(arr)
    score       = regressor.predict(emb_reduced)[0]
    return float(max(0.0, min(1.0, score)))
