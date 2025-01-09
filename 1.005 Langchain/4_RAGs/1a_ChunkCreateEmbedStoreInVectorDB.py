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
# file_path = os.path.join(current_dir, "documents", "BDM_report.txt")
file_path = os.path.join(current_dir, "documents", "alice_in_wonderland.txt")
presistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if chroma vector store already exists
if not os.path.exists(presistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    # Ensure text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load documents text content
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("\n---Document Chunks Information---")
    print(f"Number of documents chunks: {len(docs)}")
    print(f"Sample document chunk:\n{docs[0].page_content}")

    # Create embedding
    print("\n---Creating Vector Embedding---")
    embeddings = OllamaEmbeddings()  # Using the custom Ollama embeddings class
    print("\n---Finished creating Embeddings---")

    # Create vector store
    print("\n---Creating Vector Store---")
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=presistent_directory)
    print("\n---Finished creating Vector Store---")

else:
    print("Persistent directory already exists. No need to initialize")
    print("Persistent directory already exists. No need to initialize")