from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate.from_template("Explain {topic} in simple words")
prompt2 = PromptTemplate.from_template("Give 3 key points from this text:\n{text}")
prompt3 = PromptTemplate.from_template("Give real world applications of {topic}")
prompt4 = PromptTemplate.from_template("Combine the following insights:\nA:{a}\nB:{b}")

chain = (
    RunnableParallel(
        a=(prompt1 | llm | StrOutputParser() | (lambda x: {"text": x}) | prompt2 | llm | StrOutputParser()),
        b=(prompt3 | llm | StrOutputParser())
    )
    | RunnableLambda(lambda x: {"a": x["a"], "b": x["b"]})
    | prompt4
    | llm
    | StrOutputParser()
)

result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)