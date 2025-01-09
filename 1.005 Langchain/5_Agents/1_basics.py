from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, tool
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

@tool
def get_current_time():
    """Return current date and time in HH:MM:SS format."""
    return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}"

query = '''What is the current time in USA (you are in India)?'''

# PromptTemplate = PromptTemplate.from_template("{input}")
PromptTemplate = hub.pull("hwchase17/react")

tools = [get_current_time]

agent = create_react_agent(
    llm, 
    tools,
    PromptTemplate
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

result = agent_executor.invoke({"input": query})

print(result['output'])