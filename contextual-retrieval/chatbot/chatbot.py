import json
import pickle
import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer

# --- 1. LOAD ALL MODELS AND INDEXES ---
print("Loading models and indexes...")

# Embedding Model (multilingual support)
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
# Load the data
with open('contextualized_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load BM25 index
with open('indexes/bm25_index.pkl', 'rb') as f:
    bm25 = pickle.load(f)

# Load FAISS index
faiss_index = faiss.read_index('indexes/faiss_index.bin')

# Choose Ollama model (must be pulled before running: e.g. `ollama pull qwen2:7b` or `ollama create qwen3-8b -f Modelfile`)
OLLAMA_MODEL = "qwen3:8b"

print("Loading complete.")

# --- 2. DEFINE THE HYBRID SEARCH FUNCTION ---
def hybrid_search(query, k=10):
    # BM25 Search (Keyword)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # FAISS Search (Semantic)
    query_embedding = embedding_model.encode([query])
    distances, faiss_indices = faiss_index.search(
        np.array(query_embedding).astype('float32'),
        k * 5
    )  # retrieve more for fusion

    # Normalize BM25 scores
    bm25_scores_norm = (bm25_scores - np.min(bm25_scores)) / (
        np.max(bm25_scores) - np.min(bm25_scores) + 1e-8
    )
    
    # Invert FAISS distances to scores and normalize
    faiss_scores = 1 / (1 + distances[0])
    faiss_scores_norm = (faiss_scores - np.min(faiss_scores)) / (
        np.max(faiss_scores) - np.min(faiss_scores) + 1e-8
    )

    # Combine scores
    final_scores = {}
    for i, score in enumerate(bm25_scores_norm):
        final_scores[i] = final_scores.get(i, 0) + score

    for i, score in zip(faiss_indices[0], faiss_scores_norm):
        final_scores[i] = final_scores.get(i, 0) + score
        
    # Sort by combined score and get top K
    sorted_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:k]
    
    # Return the original content of the top documents
    return [data[i]['original_content'] for i in sorted_indices]

# --- 3. OLLAMA GENERATION FUNCTIONS ---
def generate_with_ollama(system_prompt, user_prompt):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        think=False
    )
    return response["message"]["content"].strip()

def stream_with_ollama(system_prompt, user_prompt):
    """Stream response from Ollama"""
    full_response = ""
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        think=False,
        stream=True
    )
    
    for chunk in stream:
        if 'message' in chunk and 'content' in chunk['message']:
            text = chunk['message']['content']
            print(text, end="", flush=True)
            full_response += text
        elif 'content' in chunk:
            text = chunk['content']
            print(text, end="", flush=True)
            full_response += text
    
    return full_response

# --- 4. MAIN CHAT LOOP ---
if __name__ == "__main__":
    print("\nJEPCO Chatbot is ready. Type 'exit' to quit.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break

        # Retrieve context
        retrieved_context = hybrid_search(user_query, k=5)
        context_str = "\n---\n".join(retrieved_context)

        # Build prompts
        system_prompt = (
            "أنت مساعد افتراضي لشركة الكهرباء الأردنية (JEPCO). "
            "أجب بدقة وباختصار اعتمادًا على المعلومات المتاحة فقط. "
            "إذا كانت المعلومات غير كافية، قل ذلك بوضوح."
        )

        user_block = f"""المعلومات المسترجعة:
---
{context_str}
---

سؤال المستخدم: {user_query}"""

        # Generate with Ollama with streaming
        try:
            print("\nJEPCO Bot: ", end="", flush=True)
            response = stream_with_ollama(system_prompt, user_block)
            print()  # Add a newline after streaming
        except Exception as e:
            print(f"An error occurred while generating the answer: {e}")
