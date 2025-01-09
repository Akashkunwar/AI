from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

PromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a fact expert who know fact about {animal}."),
        ("human","Tell me {fact_count} facts.")
    ]
)

# Create individual runnables (Steps in the chain)
format_prompt = RunnableLambda(lambda x: PromptTemplate.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = format_prompt | invoke_model | parse_output
# Any one can be used to chain both are same but pipe is easy to understand
# chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

result = chain.invoke({"animal":"cat", "fact_count":2})
print(result)
