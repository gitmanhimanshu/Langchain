from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")
class CodeFormate(BaseModel):
    language:str=Field(description="programming language of code ")
    code :str=Field(description="code of the query")
    time:str=Field(description="time complexity Of the code ")

parser=PydanticOutputParser(pydantic_object=CodeFormate)
prompt=PromptTemplate(
    template="{query}give code in fromated way {format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
    
)
chain=prompt|llm|parser
r=chain.invoke({"query":"Give prime number code "})
print(r)