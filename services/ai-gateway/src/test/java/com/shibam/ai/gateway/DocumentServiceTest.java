package com.shibam.ai.gateway;

import com.shibam.ai.gateway.dto.DocumentAnalysisRequest;
import com.shibam.ai.gateway.dto.DocumentAnalysisResponse;
import com.shibam.ai.gateway.entity.Document;
import com.shibam.ai.gateway.repository.DocumentRepository;
import com.shibam.ai.gateway.service.DocumentService;
import com.shibam.ai.gateway.service.VectorService;
import dev.langchain4j.model.chat.ChatLanguageModel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.web.multipart.MultipartFile;

import java.util.Optional;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DocumentServiceTest {

    @Mock
    private DocumentRepository documentRepository;

    @Mock
    private VectorService vectorService;

    @Mock
    private ChatLanguageModel chatModel;

    @Mock
    private MultipartFile multipartFile;

    @InjectMocks
    private DocumentService documentService;

    @BeforeEach
    void setUp() {
        when(multipartFile.getOriginalFilename()).thenReturn("test.txt");
        when(multipartFile.getInputStream()).thenReturn(
            new java.io.ByteArrayInputStream("Test document content".getBytes())
        );
    }

    @Test
    void testAnalyzeDocument_Success() throws Exception {
        // Arrange
        DocumentAnalysisRequest request = DocumentAnalysisRequest.builder()
                .file(multipartFile)
                .analysisType("full_intelligence")
                .build();

        when(chatModel.generate(anyString())).thenReturn(
            "Summary: This is a test document. Entities: Test, Document. Sentiment: Neutral"
        );
        when(documentRepository.save(any(Document.class))).thenReturn(new Document());
        doNothing().when(vectorService).storeDocumentEmbeddings(anyString(), anyString());

        // Act
        CompletableFuture<DocumentAnalysisResponse> future = documentService.analyzeDocument(request);
        DocumentAnalysisResponse response = future.get();

        // Assert
        assertNotNull(response);
        assertNotNull(response.getDocumentId());
        assertTrue(response.getDocumentId().startsWith("doc_"));
        assertEquals("neutral", response.getSentiment());
        assertTrue(response.getConfidence() > 0);

        verify(documentRepository).save(any(Document.class));
        verify(vectorService).storeDocumentEmbeddings(anyString(), anyString());
        verify(chatModel).generate(anyString());
    }

    @Test
    void testGetDocumentById_Found() {
        // Arrange
        String documentId = "doc_12345";
        Document document = Document.builder()
                .id(documentId)
                .summary("Test summary")
                .sentiment("positive")
                .confidence(0.85)
                .build();

        when(documentRepository.findById(documentId)).thenReturn(Optional.of(document));

        // Act
        Optional<DocumentAnalysisResponse> result = documentService.getDocumentById(documentId);

        // Assert
        assertTrue(result.isPresent());
        assertEquals(documentId, result.get().getDocumentId());
        assertEquals("Test summary", result.get().getSummary());
        assertEquals("positive", result.get().getSentiment());
        assertEquals(0.85, result.get().getConfidence());
    }

    @Test
    void testGetDocumentById_NotFound() {
        // Arrange
        String documentId = "doc_nonexistent";
        when(documentRepository.findById(documentId)).thenReturn(Optional.empty());

        // Act
        Optional<DocumentAnalysisResponse> result = documentService.getDocumentById(documentId);

        // Assert
        assertFalse(result.isPresent());
    }

    @Test
    void testAnalyzeDocument_ExceptionHandling() throws Exception {
        // Arrange
        DocumentAnalysisRequest request = DocumentAnalysisRequest.builder()
                .file(multipartFile)
                .analysisType("full_intelligence")
                .build();

        when(chatModel.generate(anyString())).thenThrow(new RuntimeException("AI service error"));

        // Act & Assert
        CompletableFuture<DocumentAnalysisResponse> future = documentService.analyzeDocument(request);
        
        assertThrows(RuntimeException.class, () -> {
            future.get();
        });
    }
}