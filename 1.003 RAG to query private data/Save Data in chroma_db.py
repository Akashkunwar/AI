# Load Libraries
# Define the directory where your PDFs are stored
pdf_directory = "Data"
save_dir = pdf_directory

# general
import os
import datetime

# Lancgain
from langchain_community.document_loaders import PyPDFLoader
# Langchain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Display
from IPython.display import Markdown, display

# ollama
#import ollama
#langchain
from langchain_community.document_loaders import PDFPlumberLoader

# Get a list of all PDF files in the directory

pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Initialize lists to hold pages from Nvidia and Tesla PDFs separately
nvidia_pages = []
tesla_pages = []
print(pdf_files)
# Iterate through each PDF file and load it
for pdf_file in pdf_files:
    file_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing file: {file_path}")

    # Load the PDF and split it into pages
    loader = PDFPlumberLoader(file_path=file_path)
    pages = loader.load_and_split()
    print(f"pages len={len(pages)}\n")

    # Categorize pages based on the PDF filename
    if pdf_file.startswith('NVIDIA'):
        nvidia_pages.extend(pages)
    elif pdf_file.startswith('TESLA'):
        tesla_pages.extend(pages)
    elif pdf_file.startswith('TESLA'):
        tesla_pages.extend(pages)

# print out the first page of the first document for each category as an example
if nvidia_pages:
    print("=========================================")
    print("First page of the first Nvidia document:")
    print("=========================================\n")
    print(nvidia_pages[0].page_content)
else:
    print("No Nvidia pages found in the PDFs.")



# 2.1 Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)

# Split text into chunks for Nvidia pages
nvidia_text_chunks = []
for page in nvidia_pages:
    chunks = text_splitter.split_text(page.page_content)
    nvidia_text_chunks.extend(chunks)

# Split text into chunks for Tesla pages
tesla_text_chunks = []
for page in tesla_pages:
    chunks = text_splitter.split_text(page.page_content)
    tesla_text_chunks.extend(chunks)


# 2.2 Add metadata optional


# # Example metadata management (customize as needed)
# def add_metadata(chunks, doc_title):
#     metadata_chunks = []
#     for chunk in chunks:
#         metadata = {
#             "title": doc_title,
#             "author": "company",  # Update based on document data
#             "date": str(datetime.date.today())
#         }
#         metadata_chunks.append({"text": chunk, "metadata": metadata})
#     return metadata_chunks

# # Add metadata to Nvidia chunks
# nvidia_chunks_with_metadata = add_metadata(nvidia_text_chunks, "NVIDIA Financial Report")

# # Add metadata to Tesla chunks
# tesla_chunks_with_metadata = add_metadata(tesla_text_chunks, "TESLA Financial Report")


# # 3. Create Embedding from text chunks
# import ollama

# # Function to generate embeddings for text chunks
# def generate_embeddings(text_chunks, model_name='nomic-embed-text'):
#     embeddings = []
#     for chunk in text_chunks:
#         # Generate the embedding for each chunk
#         embedding = ollama.embeddings(model=model_name, prompt=chunk)
#         embeddings.append(embedding)
#     return embeddings

# # # Example: Embed Nvidia text chunks
# # nvidia_texts = [chunk["text"] for chunk in nvidia_chunks_with_metadata]
# # nvidia_embeddings = generate_embeddings(nvidia_texts)

# # nvidia_embeddings

# # # Example: Embed Tesla text chunks
# # tesla_texts = [chunk["text"] for chunk in tesla_chunks_with_metadata]
# # tesla_embeddings = generate_embeddings(tesla_texts)

# # tesla_embeddings


# 4. Store and Use Embeddings in Chroma DB

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings

# Wrap Nvidia texts with their respective metadata into Document objects
nvidia_documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in nvidia_chunks_with_metadata]
tesla_documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in tesla_chunks_with_metadata]

# # Add Nvidia embeddings to the database
# nvidia_vector_db = Chroma.from_documents(documents=nvidia_documents,
#                       embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=False),
#                       collection_name="nvidia-local-rag")

# # Wrap Tesla texts with their respective metadata into Document objects
# tesla_documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in tesla_chunks_with_metadata]

# # # Add Tesla embeddings to the database
# # Chroma.from_documents(documents=tesla_documents,
# #                       embedding=tesla_embeddings,
# #                       collection_name="tesla-local-rag")

# # Add Tesla embeddings to the database using OllamaEmbeddings
# tesla_vector_db = Chroma.from_documents(documents=tesla_documents,
#                       embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=False),
#                       collection_name="tesla-local-rag")



# # To speed up storing embedding in chroma db
# # LLM from Ollama
# local_model = "llama3"
# llm = ChatOllama(model=local_model)

# vector_database_path = "/content/sample_data/vector_database"
# fastembedding = FastEmbedEmbeddings()

# nvidia_vector_store = Chroma.from_documents(
#         documents=nvidia_chunks,
#         embedding=fastembedding,
#         persist_directory= vector_database_path,
#         collection_name="nvidia-local-rag"
#     )

# nvidia_vector_store.persist()

# Define a directory to store the persistent database
vector_database_path = "chroma_db"

# Add Nvidia embeddings to the database with persistence
nvidia_vector_db = Chroma.from_documents(
    documents=nvidia_documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=False),
    collection_name="nvidia-local-rag",
    persist_directory=vector_database_path
)

# Persist the database after adding embeddings
nvidia_vector_db.persist()

# Add Tesla embeddings to the database with persistence
tesla_vector_db = Chroma.from_documents(
    documents=tesla_documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=False),
    collection_name="tesla-local-rag",
    persist_directory=vector_database_path
)

# Persist the database after adding embeddings
tesla_vector_db.persist()


# 5. Query Processing Multi-Query Retriever:
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# LLM from Ollama
local_model = "llama3.1"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
                                          nvidia_vector_db.as_retriever(),
                                          ChatOllama(model=local_model),
                                          prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = '''What are the main revenue drivers for Kumar Battery & Automobiles'''
# display(Markdown(chain.invoke(questions)))
# Invoke the chain with the question and store the response
response = chain.invoke(questions)
print(response)

# # Display the response in Markdown format
# if isinstance(response, str):
#     display(Markdown(response))
# else:
#     print("Unexpected response type:", type(response))

# questions = '''Can you some financial advise on Nvidia Stock to the future? should people consider buying it?'''
# display(Markdown(chain.invoke(questions)))