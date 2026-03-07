from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Use HuggingFaceEndpoint directly (no ChatHuggingFace wrapper)
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# Invoke directly
result = llm.invoke("what is the capital of India")
print(result)
