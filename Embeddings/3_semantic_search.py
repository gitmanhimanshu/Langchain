from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Initialize embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Sample documents/texts
documents = [
    "Delhi is the capital of India",
    "Python is a programming language",
    "Machine learning is a subset of artificial intelligence",
    "India is a country in South Asia",
    "JavaScript is used for web development",
    "The Taj Mahal is located in Agra, India",
    "React is a JavaScript library for building user interfaces",
    "Mumbai is the financial capital of India"
]

print("="*70)
print("STEP 1: Creating embeddings for all documents")
print("="*70)

# Create embeddings for all documents
doc_embeddings = embeddings.embed_documents(documents)

print(f"\nTotal documents: {len(documents)}")
print(f"Embedding dimension: {len(doc_embeddings[0])}")

for i, doc in enumerate(documents):
    print(f"\nDocument {i+1}: {doc}")
    print(f"  Embedding preview: [{doc_embeddings[i][0]:.4f}, {doc_embeddings[i][1]:.4f}, ...]")

# Search query
search_query = "Python is good?"

print("\n" + "="*70)
print("STEP 2: Creating embedding for search query")
print("="*70)
print(f"\nSearch Query: {search_query}")

# Create embedding for search query
query_embedding = embeddings.embed_query(search_query)
print(f"Query embedding dimension: {len(query_embedding)}")
print(f"Query embedding preview: [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ...]")

# Calculate cosine similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

print("\n" + "="*70)
print("STEP 3: Calculating similarity scores")
print("="*70)

# Calculate similarity for each document
similarities = []
for i, doc_emb in enumerate(doc_embeddings):
    similarity = cosine_similarity(query_embedding, doc_emb)
    similarities.append({
        'index': i,
        'document': documents[i],
        'similarity': similarity
    })
    print(f"\nDocument {i+1}: {documents[i]}")
    print(f"  Similarity Score: {similarity:.6f}")
    print(f"  Match Percentage: {similarity * 100:.2f}%")

# Sort by similarity (highest first)
similarities.sort(key=lambda x: x['similarity'], reverse=True)

print("\n" + "="*70)
print("STEP 4: Ranking results (Best to Worst)")
print("="*70)

for rank, item in enumerate(similarities, 1):
    print(f"\nRank {rank}:")
    print(f"  Document: {item['document']}")
    print(f"  Similarity: {item['similarity']:.6f}")
    print(f"  Match: {item['similarity'] * 100:.2f}%")
    
    # Visual representation
    bar_length = int(item['similarity'] * 50)
    bar = "█" * bar_length + "░" * (50 - bar_length)
    print(f"  Visual: [{bar}]")

print("\n" + "="*70)
print("STEP 5: Best Match Result")
print("="*70)

best_match = similarities[0]
print(f"\nQuery: {search_query}")
print(f"Best Match: {best_match['document']}")
print(f"Confidence: {best_match['similarity'] * 100:.2f}%")
