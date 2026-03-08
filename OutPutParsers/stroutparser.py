from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm=GoogleGenerativeAI(model="gemini-2.5-flash")
parser=StrOutputParser()
promt=PromptTemplate(
    template="{query}",
    input_variables=["query"]
)
chain=promt|llm|parser
r=chain.invoke("give prime number code in java")
print(r)