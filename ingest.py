import os
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
import chromadb

# Load environment variables from .env
load_dotenv()

# Check if API key is loaded
print("API key loaded:", os.getenv("GEMINI_API_KEY") is not None)

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# 1. Read PDF
# -----------------------------

PDF_PATH = "documents/research.pdf"

reader = PdfReader(PDF_PATH)

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

print("PDF text extracted successfully!")

# -----------------------------
# 2. Create chunks
# -----------------------------

def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = create_chunks(text)

print("Number of chunks:", len(chunks))

# -----------------------------
# 3. Create ChromaDB
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="research_documents"
)

# -----------------------------
# 4. Create embeddings
# -----------------------------

embeddings = []

for chunk in chunks:

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )

    embeddings.append(
        response.embeddings[0].values
    )

print("Embeddings created successfully!")

# -----------------------------
# 5. Store in ChromaDB
# -----------------------------

collection.upsert(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("Documents stored in ChromaDB successfully!")