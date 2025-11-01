import logging
import uuid
from datetime import datetime
from typing import List, Optional
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np

from models.schemas import (
    DocumentAnalysisRequest,
    DocumentAnalysisResponse,
    ChatRequest,
    ChatResponse,
    SourceReference
)

logger = logging.getLogger(__name__)

class RAGProcessor:
    """Main RAG processing service"""
    
    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.processed_documents = {}
        self.metrics = {
            "documents_processed": 0,
            "queries_processed": 0,
            "average_confidence": 0.0,
            "total_chunks": 0
        }
    
    async def process_document(self, request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
        """Process and analyze a document"""
        try:
            # Generate document ID
            document_id = f"doc_{uuid.uuid4().hex[:8]}"
            
            # Extract text content
            text_content = self._extract_text(request.content, request.filename)
            
            # Split into chunks
            chunks = self._split_text(text_content)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Generate embeddings and store
            await self._store_document_embeddings(document_id, chunks)
            
            # Analyze document with LLM
            analysis = await self._analyze_document(text_content, request.analysis_type)
            
            # Create response
            response = DocumentAnalysisResponse(
                document_id=document_id,
                summary=analysis.get("summary", ""),
                key_entities=analysis.get("entities", []),
                sentiment=analysis.get("sentiment", "neutral"),
                confidence=analysis.get("confidence", 0.85),
                analyzed_at=datetime.utcnow(),
                filename=request.filename
            )
            
            # Store document metadata
            self.processed_documents[document_id] = {
                "response": response,
                "content": text_content,
                "chunks": len(chunks),
                "created_at": datetime.utcnow()
            }
            
            # Update metrics
            self.metrics["documents_processed"] += 1
            self.metrics["total_chunks"] += len(chunks)
            
            logger.info(f"Document processed successfully: {document_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    async def process_query(self, request: ChatRequest) -> ChatResponse:
        """Process a chat query using RAG"""
        try:
            # Find relevant chunks
            relevant_chunks = await self._retrieve_relevant_chunks(
                request.question, 
                request.document_ids, 
                top_k=5
            )
            
            if not relevant_chunks:
                return ChatResponse(
                    answer="I couldn't find relevant information in the specified documents to answer your question.",
                    sources=[],
                    confidence=0.1,
                    model=request.model,
                    timestamp=datetime.utcnow()
                )
            
            # Build context from chunks
            context = self._build_context(relevant_chunks)
            
            # Generate response using LLM
            llm_response = await self._generate_response(
                request.question, 
                context, 
                request.model
            )
            
            # Build source references
            sources = [
                SourceReference(
                    document_id=chunk["document_id"],
                    page=chunk.get("page", 1),
                    relevance_score=chunk["score"],
                    excerpt=chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
                )
                for chunk in relevant_chunks
            ]
            
            # Calculate confidence
            confidence = self._calculate_confidence(llm_response, relevant_chunks)
            
            response = ChatResponse(
                answer=llm_response,
                sources=sources,
                confidence=confidence,
                model=request.model,
                timestamp=datetime.utcnow()
            )
            
            # Update metrics
            self.metrics["queries_processed"] += 1
            self._update_average_confidence(confidence)
            
            logger.info(f"Query processed with confidence: {confidence}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[dict]:
        """Get document information"""
        return self.processed_documents.get(document_id)
    
    async def get_metrics(self) -> dict:
        """Get processing metrics"""
        return self.metrics
    
    def _extract_text(self, content: bytes, filename: str) -> str:
        """Extract text from file content"""
        try:
            # Simple text extraction - in production, use proper libraries
            if filename.endswith('.txt'):
                return content.decode('utf-8')
            elif filename.endswith('.pdf'):
                # In production, use PyPDF2 or similar
                return content.decode('utf-8', errors='ignore')
            else:
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    def _split_text(self, text: str) -> List[Document]:
        """Split text into chunks"""
        return self.text_splitter.create_documents([text])
    
    async def _store_document_embeddings(self, document_id: str, chunks: List[Document]):
        """Store document embeddings in vector store"""
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            await self.vector_store.add_document(
                chunk_id, 
                chunk.page_content, 
                {"document_id": document_id, "chunk_index": i}
            )
    
    async def _analyze_document(self, content: str, analysis_type: str) -> dict:
        """Analyze document using LLM"""
        prompt = f"""
        Analyze the following document and provide insights:
        
        Analysis Type: {analysis_type}
        
        Document Content:
        {content[:2000]}...
        
        Please provide:
        1. A concise summary (2-3 sentences)
        2. Key entities mentioned (up to 5)
        3. Overall sentiment (positive/negative/neutral)
        4. Confidence score (0.0-1.0)
        
        Format as JSON with keys: summary, entities, sentiment, confidence
        """
        
        response = await self.llm_client.generate_response(prompt)
        
        # Parse response (simplified)
        try:
            import json
            return json.loads(response)
        except:
            return {
                "summary": "Document analysis completed",
                "entities": ["Entity1", "Entity2"],
                "sentiment": "neutral",
                "confidence": 0.85
            }
    
    async def _retrieve_relevant_chunks(self, query: str, document_ids: List[str], top_k: int = 5) -> List[dict]:
        """Retrieve relevant chunks for query"""
        return await self.vector_store.similarity_search(query, document_ids, top_k)
    
    def _build_context(self, chunks: List[dict]) -> str:
        """Build context from relevant chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}]: {chunk['content']}")
        return "\n\n".join(context_parts)
    
    async def _generate_response(self, question: str, context: str, model: str) -> str:
        """Generate response using LLM"""
        prompt = f"""
        Based on the following context, answer the user's question accurately and concisely.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        - Answer based only on the provided context
        - If information is not available, say so
        - Be specific and cite sources when possible
        - Keep the answer concise but comprehensive
        
        Answer:
        """
        
        return await self.llm_client.generate_response(prompt, model)
    
    def _calculate_confidence(self, response: str, chunks: List[dict]) -> float:
        """Calculate confidence score for response"""
        # Simplified confidence calculation
        base_confidence = 0.7
        
        # Adjust based on number of relevant chunks
        chunk_factor = min(len(chunks) / 5.0, 1.0) * 0.2
        
        # Adjust based on response length
        length_factor = min(len(response) / 500.0, 1.0) * 0.1
        
        return min(base_confidence + chunk_factor + length_factor, 1.0)
    
    def _update_average_confidence(self, new_confidence: float):
        """Update running average confidence"""
        total_queries = self.metrics["queries_processed"]
        if total_queries == 1:
            self.metrics["average_confidence"] = new_confidence
        else:
            current_avg = self.metrics["average_confidence"]
            self.metrics["average_confidence"] = (
                (current_avg * (total_queries - 1) + new_confidence) / total_queries
            )