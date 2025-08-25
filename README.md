# Precision Propagation Filtering (PPF)

**Precision Propagation Filtering (PPF)** is a retrieval optimization layer designed to enhance **RAG (Retrieval-Augmented Generation)** and **GraphRAG** systems by dynamically adapting retrieval precision to each query.  

The goal is simple: **send only the cleanest, most contextually relevant chunks to your LLM**, minimizing hallucinations and irrelevant context while keeping inference costs low.

---

## 🔍 Why PPF?
Most RAG pipelines suffer from **retrieval errors between 15% – 20%** (as noted in Falkordb’s 2025 benchmarks). These errors occur when:
- Too much irrelevant context is retrieved and overwhelms the model.
- Too little context is retrieved, leaving the model guessing.

GraphRAG attempted to solve this by introducing **entities and relationships** (knowledge graph retrieval). While powerful in some cases, GraphRAG can underperform when the entity graph doesn’t align with the query intent.  
Traditional RAG, when paired with **cohesive hierarchical segmentation**, can often beat GraphRAG in consistency and recall.

This is where **PPF** comes in.  

---

## ⚙️ How PPF Works
PPF adds an adaptive filtering layer between **your retriever** and **your LLM**:

1. **Predictive Precision Model**  
   - A lightweight model predicts how much precision the retrieval step should apply for a given query.  
   - Example: Some queries require **more breadth** (pull in multiple chunks), while others require **laser precision** (1–2 chunks).  

2. **Dynamic Context Control**  
   - PPF adjusts retrieval dynamically:
     - Increase context for exploratory queries.  
     - Reduce context for specific queries.  
   - Prevents "context flooding" (too many tokens) or "context starvation" (not enough tokens).  

3. **Reranking Layer**  
   - Integrates with Cohere Rerank (or any reranker you choose).  
   - Ensures only the **highest-quality chunks** are passed downstream.  

---

## 📊 Current Results
- Target hallucination rate: **< 2%**  
- Works best with **256-token chunks** (larger chunks degrade results).  
- Tested on **Neo4j GraphRAG**, moving to **standard RAG with cohesive segmentation** for broader applicability.

---

## 🚀 Features
- 🔹 **Plug-and-play**: No changes needed to your ingestion pipeline.  
- 🔹 **Model-agnostic**: Use with **any embedding model**.  
- 🔹 **Vector DB agnostic**: Works with Neo4j, Pinecone, Weaviate, Qdrant, FAISS, etc.  
- 🔹 **Logarithmic cap function**: Keeps context growth controlled, even in massive vector DBs.  
- 🔹 **Fine-tunable**: Retrain with your domain-specific data.  

---

## 🛠️ Roadmap
- [ ] Open-source the predictive model.  
- [ ] Release benchmarks on multiple datasets.  
- [ ] Expand beyond GraphRAG to cohesive RAG pipelines.  
- [ ] Optimize for long-document chunking (hierarchical segmentation).  
- [ ] Experiment with hybrid retrieval (entity graphs + segmentation).  

---

## 📚 Background Reading
- ["Enhancing RAG With Hierarchical Text Segmentation Chunking" (2025)](link-to-paper)  
- Falkordb Blog (2025): "RAG Retrieval Error Rates"  
- Neo4j GraphRAG documentation  

---

## 🤝 Contributing
I’d love feedback and contributions from anyone working on retrieval systems.  
Ideas, pull requests, and experiments are welcome!

---

## 📜 License
MIT License – Free to use, modify, and adapt.
