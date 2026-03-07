from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

# Use InferenceClient directly (simpler and more reliable)
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Use a free model that works with text generation
prompt = "what is the capital of India"

# Try with google/flan-t5-large (free and reliable)
result = client.text_generation(
    prompt,
    model="google/flan-t5-large",
    max_new_tokens=100
)

print(result)
