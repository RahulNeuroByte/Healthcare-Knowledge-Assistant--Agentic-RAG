from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    session_id: str | None = None
    query: str = Field(min_length=2, max_length=2000)


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]
    retrieved_context: list[str]
    agents_invoked: list[str]
    confidence: float
    validation_passed: bool


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    query: str = Field(min_length=2)
    rating: int = Field(ge=-1, le=1)
    correction: str | None = None
