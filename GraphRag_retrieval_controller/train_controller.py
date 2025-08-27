import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

def embedding_query(query: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

df = pd.read_csv("data/data.csv")
queries = df['Queries'].tolist()
labels = df['Label'].tolist()

embeddings = []
for q in tqdm(queries, desc="Generating embeddings"):
    embeddings.append(embedding_query(q))

X = np.array(embeddings)
y = np.array(labels)

pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.1, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

score = regressor.score(X_test, y_test)
print(f"Test R^2 Score: {score:.4f}")

joblib.dump(pca, "models/pca.joblib")
joblib.dump(regressor, "models/regression.joblib")
print("PCA and Regression models saved.")
