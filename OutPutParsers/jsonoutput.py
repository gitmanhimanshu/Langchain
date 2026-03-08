# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# parser = JsonOutputParser()

# prompt = PromptTemplate(
#     template="""
#             Give Java code for checking prime number.

#             Return response in JSON format:
#             {format_instructions}

#             Question: {query}
#             """,
#     input_variables=["query"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

# chain = prompt | llm | parser

# result = chain.invoke({"query": "Write Java code to check prime number"})

# print(result)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser=JsonOutputParser()
prompt=PromptTemplate(
    template="{query}Return response in JSON format {format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)
chain=prompt|llm|parser
r=chain.invoke({"query": "giv prime numbrer code in java"})
print(r)