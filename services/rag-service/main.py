from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from datetime import datetime
import asyncio
import os

from services.rag_processor import RAGProcessor
from services.vector_store import VectorStore
from services.llm_client import LLMClient
from models.schemas import (
    DocumentAnalysisRequest,
    DocumentAnalysisResponse,
    ChatRequest,
    ChatResponse,
    SourceReference
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation service for document intelligence",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_store = VectorStore()
llm_client = LLMClient()
rag_processor = RAGProcessor(vector_store, llm_client)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting RAG Service...")
    await vector_store.initialize()
    await llm_client.initialize()
    logger.info("RAG Service started successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/process-document", response_model=DocumentAnalysisResponse)
async def process_document(
    file: UploadFile = File(...),
    analysis_type: str = "full_intelligence"
):
    """Process and analyze uploaded document"""
    try:
        logger.info(f"Processing document: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Create request object
        request = DocumentAnalysisRequest(
            filename=file.filename,
            content=content,
            analysis_type=analysis_type
        )
        
        # Process document
        response = await rag_processor.process_document(request)
        
        logger.info(f"Document processed successfully: {response.document_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Chat with documents using RAG"""
    try:
        logger.info(f"Processing chat request: {request.question[:50]}...")
        
        # Process query using RAG
        response = await rag_processor.process_query(request)
        
        logger.info(f"Chat response generated with confidence: {response.confidence}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document information"""
    try:
        document = await rag_processor.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings/search")
async def search_embeddings(query: str, document_ids: List[str], top_k: int = 5):
    """Search for similar document chunks"""
    try:
        results = await vector_store.similarity_search(query, document_ids, top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return await rag_processor.get_metrics()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )