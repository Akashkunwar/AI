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
books_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
presistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {presistent_directory}")


# Check if chroma vector store already exists
if not os.path.exists(presistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    # Ensure text file exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"File not found: {books_dir}")

    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read text contents of each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("\n---Document Chunks Information---")
    print(f"Number of documents chunks: {len(docs)}")
    
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