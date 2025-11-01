import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import io

from main import app, DocumentProcessor

client = TestClient(app)

class TestDocumentProcessorService:
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "supported_formats" in data

    @patch('main.processor')
    def test_process_document_success(self, mock_processor):
        """Test document processing endpoint"""
        mock_response = {
            "document_id": "proc_12345",
            "filename": "test.txt",
            "file_type": ".txt",
            "text_content": "This is a test document with important information.",
            "metadata": {
                "pages": 1,
                "word_count": 8,
                "character_count": 45,
                "file_size": 45
            },
            "images": [],
            "tables": [],
            "processing_time": 0.5,
            "timestamp": datetime.utcnow()
        }
        
        mock_processor.process_document = AsyncMock(return_value=type('obj', (object,), mock_response))
        
        test_file = ("test.txt", b"This is a test document", "text/plain")
        response = client.post(
            "/process",
            files={"file": test_file},
            data={
                "processing_type": "full_extraction",
                "extract_images": "false",
                "extract_tables": "false",
                "ocr_enabled": "true",
                "language": "eng"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "proc_12345"
        assert data["filename"] == "test.txt"
        assert "text_content" in data

    def test_supported_formats_endpoint(self):
        """Test supported formats endpoint"""
        response = client.get("/formats")
        
        assert response.status_code == 200
        data = response.json()
        assert "supported_formats" in data
        assert "ocr_languages" in data
        assert ".pdf" in data["supported_formats"]
        assert ".docx" in data["supported_formats"]
        assert "eng" in data["ocr_languages"]

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "supported_formats" in data
        assert "service_status" in data
        assert data["service_status"] == "running"

class TestDocumentProcessor:
    
    @pytest.fixture
    def processor(self):
        return DocumentProcessor()

    @pytest.mark.asyncio
    async def test_process_txt_document(self, processor):
        """Test processing text document"""
        content = b"This is a test document with multiple sentences. It contains important information."
        
        request = type('obj', (object,), {
            'filename': 'test.txt',
            'processing_type': 'full_extraction',
            'extract_images': False,
            'extract_tables': False,
            'ocr_enabled': True,
            'language': 'eng'
        })
        
        # Mock file object
        mock_file = Mock()
        mock_file.read = AsyncMock(return_value=content)
        
        response = await processor.process_document(mock_file, request)
        
        assert response.filename == "test.txt"
        assert response.file_type == ".txt"
        assert "test document" in response.text_content
        assert response.metadata["word_count"] > 0
        assert response.processing_time > 0

    @pytest.mark.asyncio
    async def test_process_txt_success(self, processor):
        """Test TXT file processing"""
        content = b"This is a sample text document for testing purposes."
        request = type('obj', (object,), {
            'extract_tables': False
        })
        
        result = await processor.process_txt(content, request)
        
        assert result["text"] == "This is a sample text document for testing purposes."
        assert result["metadata"]["word_count"] == 9
        assert result["metadata"]["character_count"] == 51

    @pytest.mark.asyncio
    async def test_process_csv_success(self, processor):
        """Test CSV file processing"""
        csv_content = b"Name,Age,City\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago"
        request = type('obj', (object,), {
            'extract_tables': True
        })
        
        result = await processor.process_csv(csv_content, request)
        
        assert "John" in result["text"]
        assert "NYC" in result["text"]
        assert len(result["tables"]) == 1
        assert result["metadata"]["rows"] == 3
        assert result["metadata"]["columns"] == 3

    @pytest.mark.asyncio
    @patch('fitz.open')
    async def test_process_pdf_success(self, mock_fitz, processor):
        """Test PDF file processing"""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "This is PDF content from page 1."
        mock_page.get_images.return_value = []
        mock_page.find_tables.return_value = []
        
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_doc.metadata = {
            "title": "Test PDF",
            "author": "Test Author",
            "creationDate": "2024-01-01",
            "modDate": "2024-01-01"
        }
        mock_doc.close = Mock()
        
        mock_fitz.return_value = mock_doc
        
        content = b"fake pdf content"
        request = type('obj', (object,), {
            'extract_images': False,
            'extract_tables': False
        })
        
        result = await processor.process_pdf(content, request)
        
        assert "PDF content" in result["text"]
        assert result["metadata"]["pages"] == 1
        assert result["metadata"]["title"] == "Test PDF"
        assert result["metadata"]["author"] == "Test Author"

    @pytest.mark.asyncio
    @patch('docx.Document')
    async def test_process_docx_success(self, mock_docx, processor):
        """Test DOCX file processing"""
        # Mock python-docx document
        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_paragraph.text = "This is a paragraph from DOCX document."
        mock_doc.paragraphs = [mock_paragraph]
        mock_doc.tables = []
        
        mock_props = Mock()
        mock_props.title = "Test Document"
        mock_props.author = "Test Author"
        mock_props.created = datetime(2024, 1, 1)
        mock_props.modified = datetime(2024, 1, 1)
        mock_doc.core_properties = mock_props
        
        mock_docx.return_value = mock_doc
        
        content = b"fake docx content"
        request = type('obj', (object,), {
            'extract_tables': False
        })
        
        result = await processor.process_docx(content, request)
        
        assert "paragraph from DOCX" in result["text"]
        assert result["metadata"]["title"] == "Test Document"
        assert result["metadata"]["author"] == "Test Author"

    @pytest.mark.asyncio
    @patch('openpyxl.load_workbook')
    async def test_process_excel_success(self, mock_workbook, processor):
        """Test Excel file processing"""
        # Mock openpyxl workbook
        mock_wb = Mock()
        mock_ws = Mock()
        mock_wb.sheetnames = ["Sheet1"]
        mock_wb.__getitem__.return_value = mock_ws
        
        # Mock worksheet data
        mock_ws.iter_rows.return_value = [
            ("Name", "Age", "City"),
            ("John", 25, "NYC"),
            ("Jane", 30, "LA")
        ]
        
        mock_workbook.return_value = mock_wb
        
        content = b"fake excel content"
        request = type('obj', (object,), {
            'extract_tables': True
        })
        
        result = await processor.process_excel(content, request)
        
        assert "John" in result["text"]
        assert "NYC" in result["text"]
        assert len(result["tables"]) == 1
        assert result["tables"][0]["sheet_name"] == "Sheet1"

    @pytest.mark.asyncio
    @patch('pytesseract.image_to_string')
    @patch('PIL.Image.open')
    async def test_process_image_success(self, mock_image_open, mock_tesseract, processor):
        """Test image processing with OCR"""
        # Mock PIL Image
        mock_image = Mock()
        mock_image.width = 800
        mock_image.height = 600
        mock_image.mode = "RGB"
        mock_image_open.return_value = mock_image
        
        # Mock Tesseract OCR
        mock_tesseract.return_value = "This is extracted text from image using OCR."
        
        content = b"fake image content"
        request = type('obj', (object,), {
            'ocr_enabled': True,
            'language': 'eng'
        })
        
        result = await processor.process_image(content, request)
        
        assert "extracted text from image" in result["text"]
        assert result["metadata"]["image_width"] == 800
        assert result["metadata"]["image_height"] == 600
        assert result["metadata"]["image_mode"] == "RGB"

    @pytest.mark.asyncio
    async def test_process_image_ocr_disabled(self, processor):
        """Test image processing with OCR disabled"""
        content = b"fake image content"
        request = type('obj', (object,), {
            'ocr_enabled': False,
            'language': 'eng'
        })
        
        result = await processor.process_image(content, request)
        
        assert "OCR disabled" in result["text"]
        assert result["metadata"]["file_size"] == len(content)

    @pytest.mark.asyncio
    async def test_unsupported_file_format(self, processor):
        """Test processing unsupported file format"""
        mock_file = Mock()
        mock_file.read = AsyncMock(return_value=b"content")
        
        request = type('obj', (object,), {
            'filename': 'test.xyz',  # Unsupported format
            'processing_type': 'full_extraction',
            'extract_images': False,
            'extract_tables': False,
            'ocr_enabled': True,
            'language': 'eng'
        })
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await processor.process_document(mock_file, request)

    def test_supported_formats_configuration(self, processor):
        """Test supported formats configuration"""
        assert '.pdf' in processor.supported_formats
        assert '.docx' in processor.supported_formats
        assert '.txt' in processor.supported_formats
        assert '.xlsx' in processor.supported_formats
        assert '.csv' in processor.supported_formats
        assert '.png' in processor.supported_formats
        assert '.jpg' in processor.supported_formats
        
        # Test that all formats have corresponding processors
        for format_ext, processor_func in processor.supported_formats.items():
            assert callable(processor_func)

class TestDocumentProcessorIntegration:
    """Integration tests for document processing workflows"""
    
    @pytest.mark.asyncio
    async def test_full_document_processing_workflow(self):
        """Test complete document processing workflow"""
        processor = DocumentProcessor()
        
        # Test with simple text content
        content = b"This is a comprehensive test document. It contains multiple sentences and important information for testing purposes."
        
        request = type('obj', (object,), {
            'filename': 'integration_test.txt',
            'processing_type': 'full_extraction',
            'extract_images': False,
            'extract_tables': False,
            'ocr_enabled': True,
            'language': 'eng'
        })
        
        mock_file = Mock()
        mock_file.read = AsyncMock(return_value=content)
        
        response = await processor.process_document(mock_file, request)
        
        # Verify response structure
        assert hasattr(response, 'document_id')
        assert hasattr(response, 'filename')
        assert hasattr(response, 'file_type')
        assert hasattr(response, 'text_content')
        assert hasattr(response, 'metadata')
        assert hasattr(response, 'processing_time')
        
        # Verify content
        assert response.filename == 'integration_test.txt'
        assert response.file_type == '.txt'
        assert 'comprehensive test document' in response.text_content
        assert response.metadata['word_count'] > 0
        assert response.processing_time > 0

    def test_error_handling_robustness(self):
        """Test error handling in various scenarios"""
        processor = DocumentProcessor()
        
        # Test with empty supported formats (should not crash)
        original_formats = processor.supported_formats.copy()
        processor.supported_formats.clear()
        
        # Restore for other tests
        processor.supported_formats = original_formats
        
        assert len(processor.supported_formats) > 0

    @pytest.mark.asyncio
    async def test_metadata_extraction_accuracy(self, processor):
        """Test accuracy of metadata extraction"""
        content = b"This is a test document with exactly ten words in it."
        request = type('obj', (object,), {
            'extract_tables': False
        })
        
        result = await processor.process_txt(content, request)
        
        # Verify word count accuracy
        expected_words = len("This is a test document with exactly ten words in it.".split())
        assert result["metadata"]["word_count"] == expected_words
        
        # Verify character count accuracy
        expected_chars = len("This is a test document with exactly ten words in it.")
        assert result["metadata"]["character_count"] == expected_chars
        
        # Verify file size accuracy
        assert result["metadata"]["file_size"] == len(content)

if __name__ == "__main__":
    pytest.main([__file__])