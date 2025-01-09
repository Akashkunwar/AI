from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

AnimalFactPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a fact expert who know fact about {animal}."),
        ("human","Tell me {fact_count} facts.")
    ]
)

TranslatePromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human","Translate following text to {language}:\n{text}")
    ]
)

# Create individual runnables (Steps in the chain)
format_prompt = RunnableLambda(lambda x: AnimalFactPromptTemplate.format_prompt(**x))
prepare_for_translate = RunnableLambda(lambda output: TranslatePromptTemplate.format_prompt(
    text=output,
    language="hindi"
))


chain = format_prompt | llm | StrOutputParser() | prepare_for_translate | llm | StrOutputParser()
# Any one can be used to chain both are same but pipe is easy to understand
# chain = RunnableSequence(first=format_prompt, middle=[llm, StrOutputParser(), prepare_for_translate, llm], last=StrOutputParser())

result = chain.invoke({"animal":"cat", "fact_count":2})
print(result)
