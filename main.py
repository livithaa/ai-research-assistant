from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import app


# ==========================================
# Create FastAPI application
# ==========================================

api = FastAPI(
    title="AI Research Assistant",
    description="AI Research Assistant using RAG, ChromaDB, Gemini and LangGraph",
    version="1.0.0"
)


# ==========================================
# Request Model
# ==========================================

class QuestionRequest(BaseModel):
    question: str


# ==========================================
# Response Model
# ==========================================

class AnswerResponse(BaseModel):
    question: str
    answer: str


# ==========================================
# Home Endpoint
# ==========================================

@api.get("/")
def home():
    return {
        "message": "AI Research Assistant API is running"
    }


# ==========================================
# Health Endpoint
# ==========================================

@api.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ==========================================
# Ask Endpoint
# ==========================================

@api.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):

    # Check for empty question
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        result = app.invoke({
            "question": request.question,
            "decision": "",
            "result": ""
        })

        return {
            "question": request.question,
            "answer": result["result"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process the question: {str(e)}"
        )