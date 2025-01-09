from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import ollama
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
from typing import List
import os 
from langchain_chroma import Chroma

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)


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
presistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Load existing vector store with the embedding function
db = Chroma(persist_directory=presistent_directory, embedding_function=OllamaEmbeddings())

# User Question
query = "who is author of A Doll's House : a play"

# Retrive relevent document based on query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve the top 3 results regardless of score
)
relevent_docs = retriever.invoke(query)

# # Display the relevent result with metadata
# print("\n---Retrieved Document---")
# for i, doc in enumerate(relevent_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")

# Combine query and relevent documents
combined_input = (
    "Here are some documents what might help answer the question: \n"
    + query
    + "\n\n".join([doc.page_content for doc in relevent_docs])
    + "\n\nPlease provide a rough ansewr based on the provided documents. If answer is not found in the documnt, respond with I am not sure"
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

result = llm.invoke(messages)
print(result.content)