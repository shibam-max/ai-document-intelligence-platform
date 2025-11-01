import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

from main import app
from services.rag_processor import RAGProcessor
from services.vector_store import VectorStore
from services.llm_client import LLMClient
from models.schemas import DocumentAnalysisRequest, ChatRequest

client = TestClient(app)

class TestRAGService:
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @patch('main.rag_processor')
    def test_process_document_success(self, mock_processor):
        """Test document processing endpoint"""
        # Mock the processor response
        mock_response = {
            "document_id": "doc_12345",
            "summary": "Test document summary",
            "key_entities": ["Entity1", "Entity2"],
            "sentiment": "neutral",
            "confidence": 0.85,
            "analyzed_at": datetime.utcnow(),
            "filename": "test.txt"
        }
        
        mock_processor.process_document = AsyncMock(return_value=type('obj', (object,), mock_response))
        
        # Test file upload
        test_file = ("test.txt", b"Test document content", "text/plain")
        response = client.post(
            "/process-document",
            files={"file": test_file},
            data={"analysis_type": "full_intelligence"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.txt"

    @patch('main.rag_processor')
    def test_chat_with_documents_success(self, mock_processor):
        """Test RAG chat endpoint"""
        # Mock the processor response
        mock_response = {
            "answer": "This document discusses AI and machine learning concepts.",
            "sources": [
                {
                    "document_id": "doc_12345",
                    "page": 1,
                    "relevance_score": 0.89,
                    "excerpt": "AI and machine learning..."
                }
            ],
            "confidence": 0.92,
            "model": "gpt-3.5-turbo",
            "timestamp": datetime.utcnow()
        }
        
        mock_processor.process_query = AsyncMock(return_value=type('obj', (object,), mock_response))
        
        # Test chat request
        chat_request = {
            "question": "What is this document about?",
            "document_ids": ["doc_12345"],
            "model": "gpt-3.5-turbo"
        }
        
        response = client.post("/chat", json=chat_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert data["model"] == "gpt-3.5-turbo"

    @patch('main.rag_processor')
    def test_get_document_success(self, mock_processor):
        """Test get document endpoint"""
        mock_document = {
            "document_id": "doc_12345",
            "filename": "test.txt",
            "summary": "Test summary",
            "created_at": datetime.utcnow()
        }
        
        mock_processor.get_document = AsyncMock(return_value=mock_document)
        
        response = client.get("/documents/doc_12345")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc_12345"

    @patch('main.rag_processor')
    def test_get_document_not_found(self, mock_processor):
        """Test get document not found"""
        mock_processor.get_document = AsyncMock(return_value=None)
        
        response = client.get("/documents/nonexistent")
        
        assert response.status_code == 404

    @patch('main.vector_store')
    def test_search_embeddings_success(self, mock_vector_store):
        """Test embeddings search endpoint"""
        mock_results = [
            {
                "document_id": "doc_12345",
                "content": "Test content",
                "score": 0.89,
                "page": 1
            }
        ]
        
        mock_vector_store.similarity_search = AsyncMock(return_value=mock_results)
        
        response = client.post(
            "/embeddings/search",
            params={
                "query": "test query",
                "document_ids": ["doc_12345"],
                "top_k": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1

    @patch('main.rag_processor')
    def test_get_metrics(self, mock_processor):
        """Test metrics endpoint"""
        mock_metrics = {
            "documents_processed": 10,
            "queries_processed": 25,
            "average_confidence": 0.87,
            "total_chunks": 150
        }
        
        mock_processor.get_metrics = AsyncMock(return_value=mock_metrics)
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["documents_processed"] == 10
        assert data["queries_processed"] == 25

class TestRAGProcessor:
    
    @pytest.fixture
    def rag_processor(self):
        mock_vector_store = Mock()
        mock_llm_client = Mock()
        return RAGProcessor(mock_vector_store, mock_llm_client)

    @pytest.mark.asyncio
    async def test_process_document(self, rag_processor):
        """Test document processing logic"""
        # Mock dependencies
        rag_processor.vector_store.add_document = AsyncMock()
        rag_processor.llm_client.generate_response = AsyncMock(
            return_value='{"summary": "Test summary", "entities": ["AI", "ML"], "sentiment": "positive", "confidence": 0.85}'
        )
        
        request = DocumentAnalysisRequest(
            filename="test.txt",
            content=b"Test document content about AI and machine learning",
            analysis_type="full_intelligence"
        )
        
        response = await rag_processor.process_document(request)
        
        assert response.document_id.startswith("doc_")
        assert response.filename == "test.txt"
        assert response.confidence > 0

    @pytest.mark.asyncio
    async def test_process_query(self, rag_processor):
        """Test RAG query processing"""
        # Mock dependencies
        rag_processor._retrieve_relevant_chunks = AsyncMock(return_value=[
            {
                "document_id": "doc_12345",
                "content": "AI and machine learning content",
                "score": 0.89,
                "page": 1
            }
        ])
        rag_processor.llm_client.generate_response = AsyncMock(
            return_value="Based on the context, this document discusses AI and ML concepts."
        )
        
        request = ChatRequest(
            question="What is this document about?",
            document_ids=["doc_12345"],
            model="gpt-3.5-turbo"
        )
        
        response = await rag_processor.process_query(request)
        
        assert response.answer
        assert response.confidence > 0
        assert len(response.sources) > 0
        assert response.model == "gpt-3.5-turbo"

    def test_extract_text(self, rag_processor):
        """Test text extraction from different file types"""
        # Test plain text
        content = b"This is a test document"
        filename = "test.txt"
        
        extracted_text = rag_processor._extract_text(content, filename)
        
        assert extracted_text == "This is a test document"

    def test_split_text(self, rag_processor):
        """Test text splitting into chunks"""
        text = "This is a long document. " * 100  # Create long text
        
        chunks = rag_processor._split_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 1200 for chunk in chunks)  # Considering overlap

    def test_calculate_confidence(self, rag_processor):
        """Test confidence calculation"""
        # Test with good response
        good_response = "This is a comprehensive answer with detailed information."
        good_chunks = [{"score": 0.9}, {"score": 0.8}, {"score": 0.85}]
        
        confidence = rag_processor._calculate_confidence(good_response, good_chunks)
        
        assert 0.7 <= confidence <= 1.0
        
        # Test with poor response
        poor_response = "No info"
        poor_chunks = []
        
        confidence = rag_processor._calculate_confidence(poor_response, poor_chunks)
        
        assert confidence < 0.5

class TestVectorStore:
    
    @pytest.fixture
    def vector_store(self):
        return VectorStore()

    @pytest.mark.asyncio
    async def test_initialize(self, vector_store):
        """Test vector store initialization"""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_redis.return_value.ping = AsyncMock()
            
            with patch('sentence_transformers.SentenceTransformer') as mock_model:
                await vector_store.initialize()
                
                assert vector_store.redis_client is not None
                assert vector_store.embedding_model is not None

    @pytest.mark.asyncio
    async def test_add_document(self, vector_store):
        """Test adding document to vector store"""
        # Mock dependencies
        vector_store.embedding_model = Mock()
        vector_store.embedding_model.encode.return_value = [0.1, 0.2, 0.3]
        vector_store.redis_client = Mock()
        vector_store.redis_client.set = AsyncMock()
        vector_store.redis_client.sadd = AsyncMock()
        
        await vector_store.add_document(
            "chunk_123",
            "Test content",
            {"document_id": "doc_123", "chunk_index": 0}
        )
        
        vector_store.redis_client.set.assert_called_once()
        vector_store.redis_client.sadd.assert_called_once()

    def test_cosine_similarity(self, vector_store):
        """Test cosine similarity calculation"""
        import numpy as np
        
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 0, 0])
        
        # Test orthogonal vectors
        similarity = vector_store._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-10
        
        # Test identical vectors
        similarity = vector_store._cosine_similarity(vec1, vec3)
        assert abs(similarity - 1.0) < 1e-10

class TestLLMClient:
    
    @pytest.fixture
    def llm_client(self):
        return LLMClient()

    @pytest.mark.asyncio
    async def test_initialize(self, llm_client):
        """Test LLM client initialization"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            await llm_client.initialize()
            
            assert llm_client.api_key == 'test-key'

    @pytest.mark.asyncio
    async def test_generate_response_mock(self, llm_client):
        """Test response generation with mock"""
        llm_client.api_key = None  # Force mock response
        
        response = await llm_client.generate_response("Test prompt", "gpt-3.5-turbo")
        
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, llm_client):
        """Test sentiment analysis"""
        llm_client.api_key = None  # Force mock response
        
        result = await llm_client.analyze_sentiment("This is a great product!")
        
        assert "sentiment" in result
        assert "confidence" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]

    @pytest.mark.asyncio
    async def test_extract_entities(self, llm_client):
        """Test entity extraction"""
        llm_client.api_key = None  # Force mock response
        
        entities = await llm_client.extract_entities("John works at Microsoft in Seattle.")
        
        assert isinstance(entities, list)
        assert len(entities) > 0

if __name__ == "__main__":
    pytest.main([__file__])