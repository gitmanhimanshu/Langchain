from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Define nested structured schemas
class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")

class Contact(BaseModel):
    email: Optional[str] = Field(description="Email address", default=None)
    phone: Optional[str] = Field(description="Phone number", default=None)

class Employee(BaseModel):
    name: str = Field(description="Full name")
    position: str = Field(description="Job position")
    department: str = Field(description="Department name")
    salary: int = Field(description="Annual salary in USD")
    address: Address = Field(description="Home address")
    contact: Contact = Field(description="Contact information")
    skills: list[str] = Field(description="List of skills")

# Initialize model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_llm = llm.with_structured_output(Employee)

# Sample text
text = """
Sarah Johnson works as a Senior Software Engineer in the Engineering department. 
She earns $120,000 per year. She lives at 123 Main Street in San Francisco, USA. 
Her email is sarah.j@company.com and phone is +1-555-0123. She is skilled in 
Python, JavaScript, React, and Machine Learning.
"""

print("="*70)
print("EMPLOYEE INFORMATION EXTRACTION")
print("="*70)
print(f"\nInput Text:\n{text}")
print("\nExtracting structured data...")

# Get structured data
employee = structured_llm.invoke(f"Extract employee information: {text}")

print("\n" + "="*70)
print("EXTRACTED EMPLOYEE DATA")
print("="*70)
print(f"\n👤 Name: {employee.name}")
print(f"💼 Position: {employee.position}")
print(f"🏢 Department: {employee.department}")
print(f"💰 Salary: ${employee.salary:,}")

print(f"\n📍 Address:")
print(f"   Street: {employee.address.street}")
print(f"   City: {employee.address.city}")
print(f"   Country: {employee.address.country}")

print(f"\n📞 Contact:")
print(f"   Email: {employee.contact.email}")
print(f"   Phone: {employee.contact.phone}")

print(f"\n🛠️  Skills:")
for i, skill in enumerate(employee.skills, 1):
    print(f"   {i}. {skill}")

print("\n" + "="*70)
print("FULL OBJECT (JSON)")
print("="*70)
import json
print(json.dumps(employee.model_dump(), indent=2))
