from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)


messages = [
    SystemMessage("You are expert in customer service"),
    HumanMessage("How to mangage custoemr service?"),
]

result = llm.invoke(messages)
print(result.content)