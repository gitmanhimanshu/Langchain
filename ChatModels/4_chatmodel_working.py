from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Google Gemini works reliably and is free
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = llm.invoke("what is the capital of India")
print(result.content)
