from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_index(texts):
    """Build FAISS index from text excerpts."""
    if not texts:
        print("No texts provided, creating dummy index")
        # Return dummy components
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dummy_emb = model.encode(["dummy text"])
        idx = faiss.IndexFlatIP(dummy_emb.shape[1])
        return model, idx
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embs = model.encode(texts, normalize_embeddings=True)
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(np.array(embs, dtype="float32"))
        return model, idx
    except Exception as e:
        print(f"Error building index: {e}")
        # Return dummy components
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dummy_emb = model.encode(["dummy text"])
        idx = faiss.IndexFlatIP(dummy_emb.shape[1])
        return model, idx

def retrieve(model, idx, metas, texts, query, k=3):
    """Retrieve relevant excerpts using semantic search."""
    if not texts or not metas:
        return ["No excerpts available for context."]
    
    try:
        query_emb = model.encode([query], normalize_embeddings=True)
        scores, indices = idx.search(np.array(query_emb, dtype="float32"), min(k, len(texts)))
        
        results = []
        for i, score in zip(indices[0], scores[0]):
            if i < len(texts) and score > 0.1:  # Threshold for relevance
                results.append(f"Score: {score:.2f} - {texts[i][:200]}...")
        
        return results if results else ["No relevant excerpts found."]
    except Exception as e:
        return [f"Error during retrieval: {e}"]