from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define prompts
prompt1 = PromptTemplate(
    template="Explain {topic} in simple words",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text in 5 sentences with points:\n\n{text}",
    input_variables=["text"]
)

chain=prompt1|llm|StrOutputParser()|(lambda x:{"text":x})|prompt2|llm|StrOutputParser()


result = chain.invoke({"topic":"Artificial Intelligence"})
print(result)
print(chain.get_graph().print_ascii())


