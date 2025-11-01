package com.shibam.ai.gateway;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.shibam.ai.gateway.controller.DocumentController;
import com.shibam.ai.gateway.dto.ChatRequest;
import com.shibam.ai.gateway.dto.ChatResponse;
import com.shibam.ai.gateway.dto.DocumentAnalysisResponse;
import com.shibam.ai.gateway.service.DocumentService;
import com.shibam.ai.gateway.service.RagService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(DocumentController.class)
class DocumentControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private DocumentService documentService;

    @MockBean
    private RagService ragService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void testAnalyzeDocument_Success() throws Exception {
        // Arrange
        MockMultipartFile file = new MockMultipartFile(
                "file", 
                "test.txt", 
                "text/plain", 
                "Test document content".getBytes()
        );

        DocumentAnalysisResponse response = DocumentAnalysisResponse.builder()
                .documentId("doc_12345")
                .summary("Test document summary")
                .keyEntities(Arrays.asList("Entity1", "Entity2"))
                .sentiment("neutral")
                .confidence(0.85)
                .analyzedAt(LocalDateTime.now())
                .filename("test.txt")
                .build();

        when(documentService.analyzeDocument(any())).thenReturn(
                CompletableFuture.completedFuture(response)
        );

        // Act & Assert
        mockMvc.perform(multipart("/api/v1/documents/analyze")
                        .file(file)
                        .param("analysis_type", "full_intelligence"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.documentId").value("doc_12345"))
                .andExpect(jsonPath("$.summary").value("Test document summary"))
                .andExpect(jsonPath("$.sentiment").value("neutral"))
                .andExpect(jsonPath("$.confidence").value(0.85));
    }

    @Test
    void testChatWithDocuments_Success() throws Exception {
        // Arrange
        ChatRequest request = ChatRequest.builder()
                .question("What is this document about?")
                .documentIds(Arrays.asList("doc_12345"))
                .model("gpt-3.5-turbo")
                .build();

        ChatResponse response = ChatResponse.builder()
                .answer("This document is about testing AI capabilities.")
                .confidence(0.92)
                .model("gpt-3.5-turbo")
                .timestamp(LocalDateTime.now())
                .build();

        when(ragService.processQuery(any())).thenReturn(
                CompletableFuture.completedFuture(response)
        );

        // Act & Assert
        mockMvc.perform(post("/api/v1/ai/chat")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.answer").value("This document is about testing AI capabilities."))
                .andExpect(jsonPath("$.confidence").value(0.92))
                .andExpect(jsonPath("$.model").value("gpt-3.5-turbo"));
    }

    @Test
    void testGetDocument_Found() throws Exception {
        // Arrange
        String documentId = "doc_12345";
        DocumentAnalysisResponse response = DocumentAnalysisResponse.builder()
                .documentId(documentId)
                .summary("Test document summary")
                .sentiment("positive")
                .confidence(0.88)
                .build();

        when(documentService.getDocumentById(documentId)).thenReturn(Optional.of(response));

        // Act & Assert
        mockMvc.perform(get("/api/v1/documents/{documentId}", documentId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.documentId").value(documentId))
                .andExpect(jsonPath("$.summary").value("Test document summary"))
                .andExpect(jsonPath("$.sentiment").value("positive"));
    }

    @Test
    void testGetDocument_NotFound() throws Exception {
        // Arrange
        String documentId = "doc_nonexistent";
        when(documentService.getDocumentById(documentId)).thenReturn(Optional.empty());

        // Act & Assert
        mockMvc.perform(get("/api/v1/documents/{documentId}", documentId))
                .andExpect(status().isNotFound());
    }

    @Test
    void testHealthCheck() throws Exception {
        // Act & Assert
        mockMvc.perform(get("/api/v1/health"))
                .andExpect(status().isOk())
                .andExpect(content().string("AI Gateway is healthy"));
    }
}