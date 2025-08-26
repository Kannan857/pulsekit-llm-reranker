from typing import List, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    top_p: float = Field(0.85, ge=0.0, le=1.0)
    max_tokens: int = Field(96, ge=1, le=4096)
    stop: Optional[List[str]] = None

class ChatResponse(BaseModel):
    text: str

class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: List[str] = Field(default_factory=list)
    top_k: Optional[int] = Field(default=None, ge=1)

class RerankResponseItem(BaseModel):
    document: str
    score: float

class RerankResponse(BaseModel):
    results: List[RerankResponseItem]
