import os
import time
from llama_cpp import Llama
import numpy as np

# --- Helper Function (Cosine Similarity) ---
def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

# --- Model Initialization ---
EMBEDDING_MODEL_PATH = "./pretrainedggufmodel/mxbai-embed-large-v1-f16.gguf"  # Your model path
CTX_WINDOW_LLM = int(os.environ.get("CTX_WINDOW_LLM", 4096))
n_batch = 512

try:
    llm = Llama(model_path=EMBEDDING_MODEL_PATH, n_ctx=CTX_WINDOW_LLM, n_gpu_layers=-1, n_batch=n_batch, embedding=True)
except Exception as e:
    print(f"Error initializing Llama: {e}")
    exit()

# --- Corpus Generation ---
def generate_corpus(num_chunks, chunk_size=50):
    """Generates a corpus of text chunks."""
    corpus = []
    topics = [
        "dogs", "cats", "weather", "technology", "food", "travel", "sports",
        "music", "movies", "books", "science", "history", "politics", "art",
        "health", "finance", "education", "fashion", "nature", "relationships"
    ]
    for i in range(num_chunks):
        topic = topics[i % len(topics)]  # Cycle through topics
        text = f"This is a chunk of text about {topic}. "
        text += " ".join([f"Example sentence {j} related to {topic}." for j in range(chunk_size)])
        corpus.append(text)
    return corpus

corpus = generate_corpus(100)  # Generate 100 chunks


# --- Embed the Corpus ---
print("Embedding corpus...")
start_time = time.time()
corpus_embeddings = []
for text in corpus:
    embedding = llm.embed(text)
    corpus_embeddings.append(embedding)
end_time = time.time()
print(f"Corpus embedding time: {end_time - start_time:.2f} seconds")

# --- Test Queries ---
test_queries = [
    "What animals are known for being loyal?",
    "Tell me about different types of weather.",
    "What are some popular destinations for travel?",
    "How has technology changed over time?",
    "What is the history of the internet?",
    "Recommend some good books to read.",
    "What are the latest advancements in science?",
    ""  #Empty string
]

# --- Perform Retrieval for Each Query ---
k = 5  # Retrieve the top 5 most similar chunks

for query in test_queries:
    print(f"\nQuery: {query}")
    start_time = time.time()

    # Embed the Query
    query_embedding = llm.embed(query)

    # Calculate Similarities
    similarities = []
    for doc_embedding in corpus_embeddings:
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)

    # Retrieve Top-k
    top_k_indices = np.argsort(similarities)[::-1][:k]

    end_time = time.time()
    print(f"Retrieval time: {end_time - start_time:.4f} seconds") #Finer control

    print("\nMost similar chunks:")
    for index in top_k_indices:
        print(f"- Similarity: {similarities[index]:.4f}, Text: {corpus[index][:200]}...")  # Show first 200 chars

    # --- Optional: Correctness Check (Manual) ---
    # print("\n(Manual Check) Are these results relevant? (y/n)")
    # answer = input().strip().lower()
    # if answer != 'y':
    #     print("  (Potentially incorrect retrieval)")

llm.reset()