from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

class DocumentAnalysisRequest(BaseModel):
    """Request model for document analysis"""
    filename: str = Field(..., description="Name of the uploaded file")
    content: bytes = Field(..., description="File content as bytes")
    analysis_type: str = Field(default="full_intelligence", description="Type of analysis to perform")
    language: Optional[str] = Field(default="auto", description="Document language")
    extract_entities: bool = Field(default=True, description="Whether to extract entities")
    analyze_sentiment: bool = Field(default=True, description="Whether to analyze sentiment")
    generate_summary: bool = Field(default=True, description="Whether to generate summary")

class DocumentAnalysisResponse(BaseModel):
    """Response model for document analysis"""
    document_id: str = Field(..., description="Unique document identifier")
    summary: str = Field(..., description="Document summary")
    key_entities: List[str] = Field(default=[], description="Extracted key entities")
    sentiment: str = Field(..., description="Overall sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence score")
    analyzed_at: datetime = Field(..., description="Analysis timestamp")
    filename: str = Field(..., description="Original filename")
    analysis_type: Optional[str] = Field(description="Type of analysis performed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ChatRequest(BaseModel):
    """Request model for chat/RAG queries"""
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    document_ids: List[str] = Field(..., min_items=1, description="List of document IDs to search")
    model: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    max_tokens: int = Field(default=1000, ge=1, le=4000, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity")
    conversation_id: Optional[str] = Field(description="Conversation identifier for context")
    include_sources: bool = Field(default=True, description="Whether to include source references")

class SourceReference(BaseModel):
    """Source reference in chat responses"""
    document_id: str = Field(..., description="Source document ID")
    page: int = Field(default=1, description="Page number in document")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    excerpt: Optional[str] = Field(description="Relevant text excerpt")
    filename: Optional[str] = Field(description="Source filename")

class ChatResponse(BaseModel):
    """Response model for chat/RAG queries"""
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceReference] = Field(default=[], description="Source references")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
    model: str = Field(..., description="Model used for generation")
    timestamp: datetime = Field(..., description="Response timestamp")
    conversation_id: Optional[str] = Field(description="Conversation identifier")
    tokens_used: Optional[int] = Field(description="Number of tokens used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class EmbeddingRequest(BaseModel):
    """Request model for generating embeddings"""
    texts: List[str] = Field(..., min_items=1, description="Texts to embed")
    model: str = Field(default="text-embedding-ada-002", description="Embedding model")

class EmbeddingResponse(BaseModel):
    """Response model for embeddings"""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used")
    dimensions: int = Field(..., description="Embedding dimensions")

class SimilaritySearchRequest(BaseModel):
    """Request model for similarity search"""
    query: str = Field(..., description="Search query")
    document_ids: List[str] = Field(..., description="Documents to search in")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")

class DocumentChunk(BaseModel):
    """Document chunk model"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    metadata: dict = Field(default={}, description="Additional metadata")
    embedding: Optional[List[float]] = Field(description="Chunk embedding vector")
    page: int = Field(default=1, description="Page number")
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(default="1.0.0", description="Service version")
    dependencies: dict = Field(default={}, description="Dependency status")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MetricsResponse(BaseModel):
    """Metrics response model"""
    documents_processed: int = Field(default=0, description="Total documents processed")
    queries_processed: int = Field(default=0, description="Total queries processed")
    average_confidence: float = Field(default=0.0, description="Average confidence score")
    total_chunks: int = Field(default=0, description="Total document chunks")
    uptime_seconds: float = Field(default=0.0, description="Service uptime in seconds")
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(description="Request identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }