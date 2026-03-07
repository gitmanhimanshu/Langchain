from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()
class Structured(BaseModel):
    summery:str
    other:str = Field(
        description="occupation of person"
    )
    key_points:list[str]



llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_structured=llm.with_structured_output(Structured)
q=input("enter enput")
ans=llm_structured.invoke(q)
print(ans)
print(ans.summery)