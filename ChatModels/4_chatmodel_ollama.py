from langchain_ollama import ChatOllama

# Run Ollama locally (free, no API key needed)
llm = ChatOllama(model="llama3.2")

result = llm.invoke("what is the capital of India")
print(result.content)
