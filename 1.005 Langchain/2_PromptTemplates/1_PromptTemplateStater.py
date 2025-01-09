from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

template = '''
Write a {tone} email to {company} expressing intrest in {position} position, mentioning {skill} as key skill. Keep it within 3-4 lines.
'''

PromptTemplate = ChatPromptTemplate.from_template(template)

# prompt = PromptTemplate.format(
#     tone="professional",
#     company="Google",
#     position="Software Engineer",
#     skill="Python"
# )
# print("Format : ",prompt)

prompt = PromptTemplate.invoke({
    "tone":"professional",
    "company":"Google",
    "position":"Software Engineer",
    "skill":"Python"}   
)
print("Invoke : ",prompt)
result = llm.invoke(prompt)
print(result.content)