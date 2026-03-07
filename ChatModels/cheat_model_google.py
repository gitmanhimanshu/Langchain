from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Use Google Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
result = llm.invoke("tell me python coode for prime number")
print(result.content)
