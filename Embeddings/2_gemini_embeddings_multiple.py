from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize Google Gemini embeddings model with 32 dimensions
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="retrieval_document",
    output_dimensionality=32
)

# Generate embeddings for multiple texts
texts = [
    "what is the capital of India",
    "Delhi is the capital of India",
    "Python is a programming language"
]

# Embed multiple documents
embeddings_list = embeddings.embed_documents(texts)

print(f"Number of texts: {len(texts)}")
print(f"Embedding dimension: {len(embeddings_list[0])}")
print("\n" + "="*50)

for i, (text, embedding) in enumerate(zip(texts, embeddings_list)):
    print(f"\nText {i+1}: {text}")
    print(f"Embedding: {embedding}")
