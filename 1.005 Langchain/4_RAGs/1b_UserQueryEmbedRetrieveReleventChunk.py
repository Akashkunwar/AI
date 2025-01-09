import os
import ollama
from typing import List
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

# Custom Ollama Embeddings class
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        embeddings = []
        for text in texts:
            response = ollama.embeddings(
                model=self.model,
                prompt=text
            )
            embeddings.append(response['embedding'])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a query."""
        response = ollama.embeddings(
            model=self.model,
            prompt=text
        )
        return response['embedding']

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
presistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load existing vector store with the embedding function
db = Chroma(persist_directory=presistent_directory, embedding_function=OllamaEmbeddings())

# User Question
query = "why is Alice tired sitting on bank?"
# Alice was beginning to get very tired of sitting by her sister on the bank

# Retrive relevent document based on query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve the top 3 results regardless of score
)
relevent_docs = retriever.invoke(query)


print("\n---Retrieved Document---")
for i, doc in enumerate(relevent_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")