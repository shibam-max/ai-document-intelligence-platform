package com.shibam.ai.gateway.service;

import com.shibam.ai.gateway.dto.ChatRequest;
import com.shibam.ai.gateway.dto.ChatResponse;
import com.shibam.ai.gateway.dto.SourceReference;
import dev.langchain4j.model.chat.ChatLanguageModel;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class RagService {

    private final ChatLanguageModel chatModel;
    private final VectorService vectorService;

    public CompletableFuture<ChatResponse> processQuery(ChatRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // Step 1: Retrieve relevant document chunks using vector similarity
                List<VectorService.DocumentChunk> relevantChunks = vectorService
                        .findSimilarChunks(request.getQuestion(), request.getDocumentIds(), 5);
                
                log.info("Found {} relevant chunks for query: {}", relevantChunks.size(), request.getQuestion());
                
                // Step 2: Build context from retrieved chunks
                String context = buildContextFromChunks(relevantChunks);
                
                // Step 3: Generate RAG prompt
                String ragPrompt = buildRagPrompt(request.getQuestion(), context, request.getModel());
                
                // Step 4: Get AI response
                String systemPrompt = "You are an AI assistant that answers questions based on provided context. Always cite your sources.\n\n" + ragPrompt;
                String answer = chatModel.generate(systemPrompt);
                
                // Step 5: Build source references
                List<SourceReference> sources = relevantChunks.stream()
                        .map(chunk -> SourceReference.builder()
                                .documentId(chunk.getDocumentId())
                                .page(chunk.getPage())
                                .relevanceScore(chunk.getRelevanceScore())
                                .build())
                        .collect(Collectors.toList());
                
                // Step 6: Calculate confidence
                double confidence = calculateAnswerConfidence(answer, relevantChunks);
                
                return ChatResponse.builder()
                        .answer(answer)
                        .sources(sources)
                        .confidence(confidence)
                        .model(request.getModel())
                        .timestamp(java.time.LocalDateTime.now())
                        .build();
                        
            } catch (Exception e) {
                log.error("Error processing RAG query", e);
                throw new RuntimeException("RAG query processing failed", e);
            }
        });
    }

    private String buildContextFromChunks(List<VectorService.DocumentChunk> chunks) {
        StringBuilder context = new StringBuilder();
        context.append("Relevant document excerpts:\n\n");
        
        for (int i = 0; i < chunks.size(); i++) {
            VectorService.DocumentChunk chunk = chunks.get(i);
            context.append(String.format("[Source %d - Document: %s, Page: %d, Relevance: %.2f]\n", 
                    i + 1, chunk.getDocumentId(), chunk.getPage(), chunk.getRelevanceScore()));
            context.append(chunk.getContent());
            context.append("\n\n");
        }
        
        return context.toString();
    }

    private String buildRagPrompt(String question, String context, String model) {
        return String.format("""
            Based on the following context from documents, please answer the user's question.
            
            Context:
            %s
            
            Question: %s
            
            Instructions:
            1. Answer based only on the provided context
            2. If the context doesn't contain enough information, say so
            3. Cite specific sources when making claims
            4. Be concise but comprehensive
            5. Maintain accuracy and avoid hallucination
            
            Answer:
            """, context, question);
    }

    private double calculateAnswerConfidence(String answer, List<VectorService.DocumentChunk> chunks) {
        // Calculate confidence based on:
        // 1. Number of relevant chunks found
        // 2. Average relevance score of chunks
        // 3. Answer length and completeness
        
        if (chunks.isEmpty()) {
            return 0.1;
        }
        
        double avgRelevance = chunks.stream()
                .mapToDouble(VectorService.DocumentChunk::getRelevanceScore)
                .average()
                .orElse(0.0);
        
        double chunkCountFactor = Math.min(chunks.size() / 5.0, 1.0);
        double answerLengthFactor = Math.min(answer.length() / 500.0, 1.0);
        
        return (avgRelevance * 0.5) + (chunkCountFactor * 0.3) + (answerLengthFactor * 0.2);
    }
}