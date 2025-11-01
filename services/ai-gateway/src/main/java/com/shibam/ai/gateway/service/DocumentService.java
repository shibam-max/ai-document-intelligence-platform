package com.shibam.ai.gateway.service;

import com.shibam.ai.gateway.dto.DocumentAnalysisRequest;
import com.shibam.ai.gateway.dto.DocumentAnalysisResponse;
import com.shibam.ai.gateway.entity.Document;
import com.shibam.ai.gateway.repository.DocumentRepository;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.tika.Tika;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@Service
@RequiredArgsConstructor
@Slf4j
public class DocumentService {

    private final DocumentRepository documentRepository;
    private final VectorService vectorService;
    private final ChatLanguageModel chatModel;
    private final Tika tika = new Tika();
    
    @Value("${openai.api.key:demo-key}")
    private String openAiApiKey;

    public CompletableFuture<DocumentAnalysisResponse> analyzeDocument(DocumentAnalysisRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // Extract text from document
                String documentText = extractTextFromFile(request.getFile());
                
                // Generate document ID
                String documentId = "doc_" + UUID.randomUUID().toString().substring(0, 8);
                
                // Analyze with AI
                String analysisPrompt = buildAnalysisPrompt(documentText, request.getAnalysisType());
                String aiResponse = chatModel.generate(analysisPrompt);
                
                // Extract key information using AI
                String summary = extractSummary(aiResponse);
                List<String> keyEntities = extractEntities(documentText);
                String sentiment = analyzeSentiment(documentText);
                double confidence = calculateConfidence(aiResponse);
                
                // Store document
                Document document = Document.builder()
                        .id(documentId)
                        .filename(request.getFile().getOriginalFilename())
                        .content(documentText)
                        .summary(summary)
                        .keyEntities(keyEntities)
                        .sentiment(sentiment)
                        .confidence(confidence)
                        .analysisType(request.getAnalysisType())
                        .createdAt(LocalDateTime.now())
                        .build();
                
                documentRepository.save(document);
                
                // Store embeddings for RAG
                vectorService.storeDocumentEmbeddings(documentId, documentText);
                
                return DocumentAnalysisResponse.builder()
                        .documentId(documentId)
                        .summary(summary)
                        .keyEntities(keyEntities)
                        .sentiment(sentiment)
                        .confidence(confidence)
                        .build();
                        
            } catch (Exception e) {
                log.error("Error analyzing document", e);
                throw new RuntimeException("Document analysis failed", e);
            }
        });
    }

    public Optional<DocumentAnalysisResponse> getDocumentById(String documentId) {
        return documentRepository.findById(documentId)
                .map(this::mapToResponse);
    }

    private String extractTextFromFile(MultipartFile file) {
        try {
            // Use Apache Tika for proper text extraction
            return tika.parseToString(file.getInputStream());
        } catch (Exception e) {
            log.error("Failed to extract text from file: {}", file.getOriginalFilename(), e);
            throw new RuntimeException("Failed to extract text from file", e);
        }
    }

    private String buildAnalysisPrompt(String documentText, String analysisType) {
        return String.format("""
            Analyze the following document and provide insights based on the analysis type: %s
            
            Document Content:
            %s
            
            Please provide:
            1. A concise summary (2-3 sentences)
            2. Key entities mentioned
            3. Overall sentiment
            4. Important insights
            
            Format your response in a structured way.
            """, analysisType, documentText);
    }

    private String extractSummary(String aiResponse) {
        // Extract summary from AI response using regex or string manipulation
        String[] lines = aiResponse.split("\n");
        for (String line : lines) {
            if (line.toLowerCase().contains("summary")) {
                return line.replaceFirst("(?i).*summary:?\\s*", "").trim();
            }
        }
        return aiResponse.substring(0, Math.min(200, aiResponse.length()));
    }

    private List<String> extractEntities(String text) {
        // Simplified entity extraction - in production, use NER models
        return Arrays.asList("Entity1", "Entity2", "Entity3");
    }

    private String analyzeSentiment(String text) {
        // Simplified sentiment analysis
        String lowerText = text.toLowerCase();
        if (lowerText.contains("positive") || lowerText.contains("good") || lowerText.contains("excellent")) {
            return "positive";
        } else if (lowerText.contains("negative") || lowerText.contains("bad") || lowerText.contains("poor")) {
            return "negative";
        }
        return "neutral";
    }

    private double calculateConfidence(String response) {
        // Calculate confidence based on response quality
        if (response == null || response.trim().isEmpty()) {
            return 0.1;
        }
        
        double baseConfidence = 0.7;
        double lengthFactor = Math.min(response.length() / 500.0, 0.2);
        double structureFactor = response.contains("summary") || response.contains("entities") ? 0.1 : 0.0;
        
        return Math.min(baseConfidence + lengthFactor + structureFactor, 1.0);
    }

    private DocumentAnalysisResponse mapToResponse(Document document) {
        return DocumentAnalysisResponse.builder()
                .documentId(document.getId())
                .summary(document.getSummary())
                .keyEntities(document.getKeyEntities())
                .sentiment(document.getSentiment())
                .confidence(document.getConfidence())
                .build();
    }
}