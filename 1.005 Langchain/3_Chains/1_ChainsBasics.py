from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

PromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a fact expert who know fact about {animal}."),
        ("human","Tell me {fact_count} facts.")
    ]
)

chain = PromptTemplate | llm | StrOutputParser()

result = chain.invoke({"animal":"cat", "fact_count":2})
print(result)
