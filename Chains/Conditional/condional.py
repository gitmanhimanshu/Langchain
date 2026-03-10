from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
mathprompt=PromptTemplate(
    template="solve this problem {topic}",
    input_variables=['topic']
)
generalprompt=PromptTemplate(
    template="Give answer for {topic}",
    input_variables=['topic']
)
math=mathprompt|llm|StrOutputParser()
general=generalprompt|llm|StrOutputParser()
chain=RunnableBranch(
    (lambda x:'math' in x['topic'],math),
    general
)

ans=chain.invoke({"topic":"math 5+6"})
print(ans)
print(chain.get_graph().print_ascii())