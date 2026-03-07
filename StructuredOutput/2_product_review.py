from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Define structured schema for product review analysis
class ProductReview(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    rating: int = Field(
        description="Rating from 1 to 5 stars",
        ge=1,
        le=5
    )
    summary: str = Field(
        description="Brief summary of the review in one sentence"
    )
    pros: list[str] = Field(
        description="List of positive points mentioned"
    )
    cons: list[str] = Field(
        description="List of negative points mentioned"
    )
    would_recommend: bool = Field(
        description="Whether the reviewer would recommend this product"
    )

# Initialize model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_llm = llm.with_structured_output(ProductReview)

# Sample review
review_text = """
I bought this laptop last month and I'm really impressed! The battery life is 
amazing - lasts all day. The screen is bright and clear. Performance is great 
for coding and video editing. However, it's a bit heavy to carry around and 
the keyboard could be better. The trackpad is also not very responsive. 
Overall, I'm happy with my purchase and would recommend it to developers.
"""

print("="*70)
print("PRODUCT REVIEW ANALYSIS")
print("="*70)
print(f"\nReview Text:\n{review_text}")
print("\nAnalyzing...")

# Get structured analysis
result = structured_llm.invoke(f"Analyze this product review: {review_text}")

print("\n" + "="*70)
print("STRUCTURED ANALYSIS")
print("="*70)
print(f"\nSentiment: {result.sentiment.upper()}")
print(f"Rating: {'⭐' * result.rating} ({result.rating}/5)")
print(f"\nSummary: {result.summary}")
print(f"\nPros:")
for i, pro in enumerate(result.pros, 1):
    print(f"  {i}. ✓ {pro}")
print(f"\nCons:")
for i, con in enumerate(result.cons, 1):
    print(f"  {i}. ✗ {con}")
print(f"\nWould Recommend: {'Yes ✓' if result.would_recommend else 'No ✗'}")

print("\n" + "="*70)
print("RAW DATA (JSON)")
print("="*70)
import json
print(json.dumps(result.model_dump(), indent=2))
