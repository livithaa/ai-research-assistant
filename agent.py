import os
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
import chromadb

from langgraph.graph import StateGraph, START, END

from tools import calculator


# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ==========================================
# 2. Connect to ChromaDB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="research_documents"
)


# ==========================================
# 3. Define Agent State
# ==========================================

class AgentState(TypedDict):
    question: str
    decision: str
    result: str


# ==========================================
# 4. Decision Node
# ==========================================

def decision_node(state: AgentState):

    question = state["question"]

    prompt = f"""
You are deciding which capability should handle a user's question.

Return ONLY one word:

calculator
or
research

Use calculator for mathematical calculations.

Use research for questions about the research document.

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    decision = response.text.strip().lower()

    if "calculator" in decision:
        decision = "calculator"
    else:
        decision = "research"

    print("Selected tool:", decision)

    return {
        "decision": decision
    }


# ==========================================
# 5. Calculator Node
# ==========================================

def calculator_node(state: AgentState):

    question = state["question"]

    prompt = f"""
Extract the mathematical operation from this question.

Return ONLY in this exact format:

number1,number2,operation

Allowed operations:
add
subtract
multiply
divide

Examples:

Question: 50 * 50
Answer: 50,50,multiply

Question: 25 + 10
Answer: 25,10,add

Question: 100 / 4
Answer: 100,4,divide

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    values = response.text.strip().split(",")

    a = float(values[0].strip())
    b = float(values[1].strip())
    operation = values[2].strip().lower()

    result = calculator(a, b, operation)

    return {
        "result": f"Calculator result: {result}"
    }


# ==========================================
# 6. Research / RAG Node
# ==========================================

def research_node(state: AgentState):

    question = state["question"]

    # Convert question into embedding
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    query_embedding = response.embeddings[0].values

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    documents = results["documents"][0]

    # Combine retrieved chunks
    context = "\n\n".join(documents)

    # Ask Gemini using retrieved context
    prompt = f"""
You are an AI research assistant.

Answer the user's question using only the
information provided in the research context.

If the answer is not present in the context,
say:

"I could not find the answer in the provided document."

Research context:
{context}

Question:
{question}
"""

    answer_response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return {
        "result": answer_response.text
    }


# ==========================================
# 7. Choose Next Node
# ==========================================

def choose_next(state: AgentState):

    return state["decision"]


# ==========================================
# 8. Build LangGraph
# ==========================================

graph = StateGraph(AgentState)

graph.add_node("decision", decision_node)
graph.add_node("calculator", calculator_node)
graph.add_node("research", research_node)

graph.add_edge(START, "decision")

graph.add_conditional_edges(
    "decision",
    choose_next,
    {
        "calculator": "calculator",
        "research": "research"
    }
)

graph.add_edge("calculator", END)
graph.add_edge("research", END)


# ==========================================
# 9. Compile Agent
# ==========================================

app = graph.compile()


# ==========================================
# IMPORTANT:
# No input() here!
#
# FastAPI will send the question to app.invoke()
# ==========================================