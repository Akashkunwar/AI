from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
import time

# Define the path to the Chroma DB directory
vector_database_path = "chroma_db"

# Initialize the embedding function
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# Load the Chroma DB with the embedding function
nvidia_vector_db = Chroma(
    collection_name="nvidia-local-rag",
    persist_directory=vector_database_path,
    embedding_function=embedding_function
)

tesla_vector_db = Chroma(
    collection_name="tesla-local-rag",
    persist_directory=vector_database_path,
    embedding_function=embedding_function
)

# Define the LLM model
local_model = "llama3.1"
llm = ChatOllama(model=local_model)

# Define the query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Create retriever for the Chroma DB
retriever = MultiQueryRetriever.from_llm(
    nvidia_vector_db.as_retriever(),
    ChatOllama(model=local_model),
    prompt=QUERY_PROMPT
)

# Define the RAG prompt template
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

# Create the prompt chain
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def stream_response(chain, question):
    # Prepare the input for the chain
    input_data = {"context": "", "question": question}
    
    # Start streaming
    print("AI: ", end="")
    
    # Measure start time
    start_time = time.time()
    
    response_chunks = chain.stream(input_data)
    
    for chunk in response_chunks:
        print(chunk, end="", flush=True)
    
    # Measure end time
    end_time = time.time()
    
    print()  # New line after streaming the full response
    print(f"Response Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    question = "what is detail of the person who create this report?"
    stream_response(chain, question)


# from langchain_chroma import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_core.output_parsers import StrOutputParser

# # Define the path to the Chroma DB directory
# vector_database_path = "chroma_db"

# # Initialize the embedding function
# embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# # Load the Chroma DB with the embedding function
# nvidia_vector_db = Chroma(
#     collection_name="nvidia-local-rag",
#     persist_directory=vector_database_path,
#     embedding_function=embedding_function
# )

# tesla_vector_db = Chroma(
#     collection_name="tesla-local-rag",
#     persist_directory=vector_database_path,
#     embedding_function=embedding_function
# )

# # Define the LLM model
# local_model = "llama3.1"
# llm = ChatOllama(model=local_model)

# # Define the query prompt template
# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from
#     a vector database. By generating multiple perspectives on the user question, your
#     goal is to help the user overcome some of the limitations of the distance-based
#     similarity search. Provide these alternative questions separated by newlines.
#     Original question: {question}""",
# )

# # Create retriever for the Chroma DB
# retriever = MultiQueryRetriever.from_llm(
#     nvidia_vector_db.as_retriever(),
#     ChatOllama(model=local_model),
#     prompt=QUERY_PROMPT
# )

# # Define the RAG prompt template
# template = """Answer the question based ONLY on the following context:
# {context}
# Question: {question}
# """

# # Create the prompt chain
# prompt = ChatPromptTemplate.from_template(template)
# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# def stream_response(chain, question):
#     # Prepare the input for the chain
#     input_data = {"context": "", "question": question}
    
#     # Start streaming
#     print("AI: ", end="")
#     response_chunks = chain.stream(input_data)
    
#     for chunk in response_chunks:
#         print(chunk, end="", flush=True)
#     print()  # New line after streaming the full response

# if __name__ == "__main__":
#     question = "What are the main revenue drivers for Kumar Battery & Automobiles"
#     stream_response(chain, question)
