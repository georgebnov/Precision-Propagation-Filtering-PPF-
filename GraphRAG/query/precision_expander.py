from typing import List, Tuple
from embedding_query import embedding_query
from cosine_similarity import cosine_similarity

def process_paths_for_ppf(
    query_embedding: List[float],
    paths: List[Tuple[List[str], List[str], List[List[float]]]]
) -> List[Tuple[List[str], List[str], float]]:
    enriched_chains = []

    for ids_chain, contents_chain, embeddings_chain in paths:
        #Compute the element-wise average of embeddings in the path
        combined_embedding = [
            sum(values) / len(values)
            for values in zip(*embeddings_chain)
        ]

        #Compute the average of combined path embedding and query embedding
        averaged_embedding = [
            (q + c) / 2
            for q, c in zip(query_embedding, combined_embedding)
        ]

        # Compute cosine similarity
        precision = cosine_similarity(query_embedding, averaged_embedding)

        # Append the structured enriched chain
        enriched_chains.append((ids_chain, contents_chain, precision))

    print(f"\nProcessed {len(enriched_chains)} enriched chains with computed precision using query+path averaging.")
    return enriched_chains

def filter_by_precision(chains: List[tuple], threshold: float = 0.8, top_n: int = 5) -> List[tuple]:
    """
    Filter structured chains by precision threshold, fallback to top-N if none pass.
    Each item in `chains`:
        (List[str] ids_chain, List[str] contents_chain, float precision)
    """
    filtered = [chain for chain in chains if chain[2] >= threshold]

    if not filtered:
        # fallback to top-N
        chains_sorted = sorted(chains, key=lambda x: x[2], reverse=True)
        return chains_sorted[:top_n]

    return filtered[:top_n]

def build_llm_context_from_chains(filtered_chains: List[tuple]) -> str:
    """
    Build LLM-ready context from filtered chains.
    Concatenates the text of each chain into coherent context blocks.
    """
    context_blocks = []
    for ids_chain, contents_chain, precision in filtered_chains:
        block = " ".join(contents_chain)
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

