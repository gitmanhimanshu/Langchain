from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Define structured output schema using Pydantic
class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    city: str = Field(description="City where person lives")
    occupation: str = Field(description="Person's job or profession")

# Initialize model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create structured model
structured_llm = llm.with_structured_output(PersonInfo)

# Test with a prompt
prompt = "John Smith is a 35 year old software engineer living in New York"

print("="*60)
print("STRUCTURED OUTPUT EXAMPLE")
print("="*60)
print(f"\nInput: {prompt}")
print("\nProcessing...")

# Get structured response
result = structured_llm.invoke(prompt)

print("\n" + "="*60)
print("STRUCTURED OUTPUT (Pydantic Object)")
print("="*60)
print(f"\nType: {type(result)}")
print(f"\nResult:\n{result}")

print("\n" + "="*60)
print("ACCESSING INDIVIDUAL FIELDS")
print("="*60)
print(f"Name: {result.name}")
print(f"Age: {result.age}")
print(f"City: {result.city}")
print(f"Occupation: {result.occupation}")

print("\n" + "="*60)
print("AS DICTIONARY")
print("="*60)
print(result.model_dump())
