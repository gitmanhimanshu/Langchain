from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Use HuggingFaceEndpoint directly
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

# Invoke directly
result = llm.invoke("what is the capital of India")
print(result)
