# app/models.py
from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    prompt: str
    session_id: str # To maintain conversation history later

class ChatResponse(BaseModel):
    response: str

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResponse(BaseModel):
    ranked_documents: List[str]