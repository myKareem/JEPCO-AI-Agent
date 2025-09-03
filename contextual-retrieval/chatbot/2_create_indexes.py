import json
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

print("Loading contextualized data...")
try:
    with open('contextualized_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: contextualized_data.json not found. Run 1_preprocess_data.py first.")
    exit()

# Get the text content we will be indexing
corpus = [item['contextualized_content'] for item in data]
print(f"Loaded {len(corpus)} documents to index.")

# --- Create BM25 Index (for keyword search) ---
print("\nCreating BM25 index...")
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Save the BM25 index to a file
with open('indexes/bm25_index.pkl', 'wb') as f:
    pickle.dump(bm25, f)
print("BM25 index created and saved to indexes/bm25_index.pkl")


# --- Create FAISS Index (for semantic search) ---
print("\nCreating FAISS index...")
# We use a multilingual model, which is great for Arabic
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

print("Generating embeddings for the corpus... (This might take a while)")
embeddings = embedding_model.encode(corpus, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Create a FAISS index
index_dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(index_dimension)
faiss_index.add(embeddings)

# Save the FAISS index
faiss.write_index(faiss_index, 'indexes/faiss_index.bin')
print("FAISS index created and saved to indexes/faiss_index.bin")

print("\nAll indexes created successfully.")