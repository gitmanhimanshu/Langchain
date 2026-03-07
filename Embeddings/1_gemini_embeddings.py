from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize Google Gemini embeddings model with custom dimension
# Supported dimensions: 768, 1536, 3072 (default), or custom values
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="retrieval_document",
    output_dimensionality=32
)

# Generate embedding for a single text
text = "what is the capital of India"
embedding = embeddings.embed_query(text)

print(f"Text: {text}")
print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
print(f"\nFull embedding:\n{embedding}")
