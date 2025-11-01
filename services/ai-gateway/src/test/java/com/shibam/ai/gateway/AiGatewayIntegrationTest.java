package com.shibam.ai.gateway;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.shibam.ai.gateway.dto.ChatRequest;
import com.shibam.ai.gateway.entity.Document;
import com.shibam.ai.gateway.repository.DocumentRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureWebMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Arrays;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureWebMvc
@ActiveProfiles("test")
@Transactional
class AiGatewayIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private DocumentRepository documentRepository;

    @Autowired
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        documentRepository.deleteAll();
    }

    @Test
    void testFullDocumentProcessingWorkflow() throws Exception {
        // Step 1: Upload and analyze document
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "integration-test.txt",
                "text/plain",
                "This is a comprehensive integration test document with positive sentiment and important entities like AI, Machine Learning, and Spring Boot.".getBytes()
        );

        String documentAnalysisResponse = mockMvc.perform(multipart("/api/v1/documents/analyze")
                        .file(file)
                        .param("analysis_type", "full_intelligence"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.documentId").exists())
                .andExpect(jsonPath("$.summary").exists())
                .andExpect(jsonPath("$.sentiment").exists())
                .andExpect(jsonPath("$.confidence").exists())
                .andReturn()
                .getResponse()
                .getContentAsString();

        // Extract document ID from response
        com.fasterxml.jackson.databind.JsonNode responseNode = objectMapper.readTree(documentAnalysisResponse);
        String documentId = responseNode.get("documentId").asText();

        // Step 2: Retrieve document by ID
        mockMvc.perform(get("/api/v1/documents/{documentId}", documentId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.documentId").value(documentId))
                .andExpect(jsonPath("$.summary").exists());

        // Step 3: Chat with the document using RAG
        ChatRequest chatRequest = ChatRequest.builder()
                .question("What technologies are mentioned in this document?")
                .documentIds(Arrays.asList(documentId))
                .model("gpt-3.5-turbo")
                .maxTokens(500)
                .temperature(0.7)
                .build();

        mockMvc.perform(post("/api/v1/ai/chat")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(chatRequest)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.answer").exists())
                .andExpect(jsonPath("$.confidence").exists())
                .andExpect(jsonPath("$.model").value("gpt-3.5-turbo"));

        // Step 4: Verify document was stored in database
        Document storedDocument = documentRepository.findById(documentId).orElse(null);
        assert storedDocument != null;
        assert storedDocument.getFilename().equals("integration-test.txt");
        assert storedDocument.getContent().contains("integration test document");
    }

    @Test
    void testHealthEndpoint() throws Exception {
        mockMvc.perform(get("/api/v1/health"))
                .andExpect(status().isOk())
                .andExpect(content().string("AI Gateway is healthy"));
    }

    @Test
    void testDocumentNotFound() throws Exception {
        mockMvc.perform(get("/api/v1/documents/nonexistent-doc"))
                .andExpect(status().isNotFound());
    }

    @Test
    void testInvalidChatRequest() throws Exception {
        ChatRequest invalidRequest = ChatRequest.builder()
                .question("") // Empty question
                .documentIds(Arrays.asList())  // Empty document list
                .build();

        mockMvc.perform(post("/api/v1/ai/chat")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(invalidRequest)))
                .andExpect(status().isBadRequest());
    }

    @Test
    void testUnsupportedFileType() throws Exception {
        MockMultipartFile unsupportedFile = new MockMultipartFile(
                "file",
                "test.xyz",
                "application/unknown",
                "Unsupported file content".getBytes()
        );

        mockMvc.perform(multipart("/api/v1/documents/analyze")
                        .file(unsupportedFile))
                .andExpect(status().isInternalServerError());
    }
}