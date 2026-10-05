import os
from dotenv import load_dotenv
from google import genai
import chromadb

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="research_documents"
)

# -----------------------------
# 1. Ask a question
# -----------------------------

question = input("Ask a question about the PDF: ")

# -----------------------------
# 2. Convert question to embedding
# -----------------------------

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

query_embedding = response.embeddings[0].values

# -----------------------------
# 3. Retrieve relevant chunks
# -----------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

documents = results["documents"][0]

# Combine retrieved chunks
context = "\n\n".join(documents)

# -----------------------------
# 4. Send context + question to Gemini
# -----------------------------

prompt = f"""
You are an AI research assistant.

Answer the user's question using only the information
provided in the context below.

If the answer is not present in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}
"""

answer_response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

# -----------------------------
# 5. Display final answer
# -----------------------------

print("\nAI Answer:")
print(answer_response.text)