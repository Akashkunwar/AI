# import ollama

# response = ollama.generate(model='llama3.1',
# prompt='what is a qubit?')
# print(response['response'])

# import ollama
# response = ollama.chat(model='mistral', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])

# import ollama

# stream = ollama.chat(
#     model='llama3.1',
#     messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
#     stream=True,
# )

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)

# import ollama
# print(ollama.generate(model='llama3.1', prompt='Why is the sky blue?'))


# # import ollama

# # print(ollama.embeddings(model='nomic-embed-text', prompt="How are you"))








import os
import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings

# Directory containing the PDFs
pdf_directory = "Data"

# Get a list of all PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Process each PDF file
def process_pdf(file_path):
    print(f"Processing file: {file_path}")
    
    # Load the PDF and split it into pages
    loader = PDFPlumberLoader(file_path=file_path)
    pages = loader.load_and_split()
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    
    # Split the pages into chunks
    text_chunks = []
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        text_chunks.extend(chunks)
    
    # Add metadata (using filename as title)
    file_title = os.path.basename(file_path).replace('.pdf', '')
    chunks_with_metadata = add_metadata(text_chunks, file_title)
    
    # Wrap chunks with metadata into Document objects
    documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in chunks_with_metadata]
    
    return documents

# Function to add metadata to chunks
def add_metadata(chunks, doc_title):
    metadata_chunks = []
    for chunk in chunks:
        metadata = {
            "title": doc_title,
            "author": "company",  # Customize as needed
            "date": str(datetime.date.today())
        }
        metadata_chunks.append({"text": chunk, "metadata": metadata})
    return metadata_chunks

# Process and store documents in Chroma for all PDF files
def store_embeddings_in_chroma(pdf_files):
    all_documents = []
    
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_directory, pdf_file)
        documents = process_pdf(file_path)
        all_documents.extend(documents)
    
    # Add all documents to the Chroma vector store
    vector_db = Chroma.from_documents(documents=all_documents,
                                      embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=False),
                                      collection_name="all-documents-rag")
    return vector_db

# Store embeddings for all PDFs in the directory
vector_db = store_embeddings_in_chroma(pdf_files)

# 5. Query Processing using Multi-Query Retriever
local_model = "llama3.1"
llm = ChatOllama(model=local_model)

# Define the Multi-Query Retriever
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
                                          vector_db.as_retriever(),
                                          ChatOllama(model=local_model),
                                          prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = PromptTemplate.from_template(template)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query example
questions = '''give me brief summary of the data and guess data is about what and what is purpose of it?'''
response = chain.invoke(questions)
print(response)
