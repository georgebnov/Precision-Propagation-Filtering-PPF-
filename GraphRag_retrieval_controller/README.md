# GraphRAG Retrieval Controller

A lightweight, open-source retrieval controller that classifies user queries on a spectrum from precise to broad, dynamically adjusting GraphRAG hops and retrieval chunk windows using a simple linear regression model.

## Features
- Uses OpenAI embeddings for semantic input  
- Trains a PCA + LinearRegression pipeline for generality scoring  
- Dynamically outputs `hops` and `k_chunks` for GraphRAG retrieval

## Usage

### 1 Install dependencies
pip install -r requirements.txt

### 2 Prepare your data
Add labeled queries to `data/labeled_queries.csv`.

### 3 Train the model
python train_controller.py

### 4 Test live predictions
python predict_controller.py

## Contributing
Pull requests welcome for adding:
- New models (logistic regression, classification options)
- Larger datasets
- Evaluation metrics
