from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Define schema for general text analysis
class TextAnalysis(BaseModel):
    topic: str = Field(description="Main topic of the text")
    category: Literal["technology", "business", "health", "education", "entertainment", "other"] = Field(
        description="Category of the content"
    )
    key_points: list[str] = Field(description="3-5 key points from the text")
    entities: list[str] = Field(description="Important names, places, or organizations mentioned")
    tone: Literal["formal", "informal", "neutral", "technical"] = Field(
        description="Writing tone"
    )
    word_count_estimate: int = Field(description="Estimated word count")

# Initialize model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_llm = llm.with_structured_output(TextAnalysis)

print("="*70)
print("INTERACTIVE TEXT ANALYZER")
print("="*70)
print("Enter text to analyze (type 'exit' to quit)")
print("="*70)

while True:
    print("\n" + "-"*70)
    user_text = input("\nEnter text: ")
    
    if user_text.lower() == "exit":
        print("\nGoodbye!")
        break
    
    if not user_text.strip():
        print("Please enter some text!")
        continue
    
    print("\nAnalyzing...")
    
    # Get structured analysis
    result = structured_llm.invoke(f"Analyze this text: {user_text}")
    
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    print(f"\n📌 Topic: {result.topic}")
    print(f"📂 Category: {result.category.upper()}")
    print(f"📝 Tone: {result.tone.capitalize()}")
    print(f"📊 Word Count: ~{result.word_count_estimate} words")
    
    print(f"\n🔑 Key Points:")
    for i, point in enumerate(result.key_points, 1):
        print(f"   {i}. {point}")
    
    print(f"\n🏷️  Entities:")
    for entity in result.entities:
        print(f"   • {entity}")
