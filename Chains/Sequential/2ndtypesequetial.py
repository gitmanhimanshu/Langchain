from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()
llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt1=PromptTemplate(
    template="Explain {topic} in simple words",
    input_variables=["topic"]
)
prompt2=PromptTemplate(
    template="Give real Life Example of  {topic} ",
    input_variables=["topic"]
)
chain=RunnableParallel(
    explanation=prompt1|llm|StrOutputParser(),
    applications=prompt2|llm|StrOutputParser()
)
ans=chain.invoke({"topic":"Artificial Intelligence"})
print(ans)
print(chain.get_graph().print_ascii())
