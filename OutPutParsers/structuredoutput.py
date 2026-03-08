from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser ,ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define response schemas
response_schemas = [
    ResponseSchema(
        name="language", 
        description="The programming language of the code"
    ),
    ResponseSchema(
        name="code", 
        description="Code to check if a number is prime"
    ),
    ResponseSchema(
        name="time_complexity",
        description="Time complexity of the code"
    ),
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get format instructions
format_instructions = parser.get_format_instructions()

# Create prompt
prompt = PromptTemplate(
    template="{query}\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions}
)

# Create chain
chain = prompt | llm | parser

# Invoke
result = chain.invoke({"query": "Write a Python function to check if a number is prime"})

print("="*60)
print("STRUCTURED OUTPUT RESULT")
print("="*60)
print(f"\nLanguage: {result['language']}")
print(f"\nCode:\n{result['code']}")
print(f"\nTime Complexity: {result['time_complexity']}")
