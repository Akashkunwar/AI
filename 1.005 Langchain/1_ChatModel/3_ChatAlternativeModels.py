from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# Antropic
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)

# Google Generative AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

result = llm.invoke("Hello, how are you?")
print(result.content)