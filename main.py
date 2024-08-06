from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime


from rag_engine import RAGEngine

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

rag_engine = RAGEngine()

from pydantic import BaseModel
from typing import List, Literal


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryMessage(BaseModel):
    query: str
    llm: str
    namespace: str = "scbx_hr"
    engine: Literal["chat", "chat-v2", "query", "retrieve"] = "chat"
    system_prompt: str
    messages: List[Message]


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str


@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="ok", timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


@app.get("/query")
def test_query(q: str, model: str):
    result = rag_engine.test_query(q, model)
    return result


@app.post("/messages")
def query_messages(data: QueryMessage):
    result = rag_engine.query_messages(data)
    return result
