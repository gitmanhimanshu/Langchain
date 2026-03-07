from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("="*50)
print("Simple Chat App")
print("Type 'exit' to quit")
print("="*50)

# while True:
#     # Get user input
#     user_input = input("\nYou: ")
    
#     # Check if user wants to exit
#     if user_input.lower() == "exit":
#         print("\nGoodbye!")
#         break
    
#     # Get response from model
#     response = llm.invoke(user_input)
    
#     # Print response
#     print(f"\nBot: {response.content}")

while True:
    q=input("enter input")
    if q.lower()=='exit':
        break
    result=llm.invoke(q)
    print(result.content)