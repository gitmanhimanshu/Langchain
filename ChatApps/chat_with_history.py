from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Store chat history
chat_history = []

print("="*50)
print("Chat App with History")
print("Type 'exit' to quit")
print("="*50)

while True:
    # Get user input
    user_input = input("\nYou: ")
    
    # Check if user wants to exit
    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break
    
    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))
    
    # Get response from model with full history
    response = llm.invoke(chat_history)
    
    # Add AI response to history
    chat_history.append(AIMessage(content=response.content))
    
    # Print response
    print(f"\nBot: {response.content}")
