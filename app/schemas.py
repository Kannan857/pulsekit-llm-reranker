from typing import List, Literal, Optional
from pydantic import BaseModel, Field

Channel = Literal["voice", "chat"]

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class GenerationParams(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None

class ChannelDefaults(BaseModel):
    voice: Optional[GenerationParams] = None
    chat: Optional[GenerationParams] = None

class ChatRequest(BaseModel):
    prompt: str
    channel: Channel = Field(default="chat")
    # OPTIONAL per-request overrides (highest precedence)
    params: Optional[GenerationParams] = None
    # OPTIONAL per-channel defaults sent by the caller (middle precedence)
    defaults: Optional[ChannelDefaults] = None
    # OPTIONAL conversation/RAG inputs
    history: Optional[List[Message]] = None
    context: Optional[str] = None
    # OPTIONAL: will be forwarded to vLLM "user" field for logging/limits
    user: Optional[str] = None

class ChatResponse(BaseModel):
    text: str

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 5

class RerankResponseItem(BaseModel):
    document: str
    score: float

class RerankResponse(BaseModel):
    results: List[RerankResponseItem]
