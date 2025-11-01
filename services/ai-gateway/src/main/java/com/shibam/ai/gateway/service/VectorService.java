package com.shibam.ai.gateway.service;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

@Service
@RequiredArgsConstructor
@Slf4j
public class VectorService {

    private final EmbeddingModel embeddingModel;
    private final RedisTemplate<String, Object> redisTemplate;
    
    private static final String VECTOR_KEY_PREFIX = "vector:";
    private static final String DOCUMENT_KEY_PREFIX = "doc:";
    private static final int CHUNK_SIZE = 1000;
    private static final int CHUNK_OVERLAP = 200;

    public void storeDocumentEmbeddings(String documentId, String content) {
        try {
            // Split document into chunks
            List<String> chunks = splitIntoChunks(content);
            log.info("Split document {} into {} chunks", documentId, chunks.size());
            
            // Generate embeddings for each chunk
            for (int i = 0; i < chunks.size(); i++) {
                String chunkId = documentId + "_chunk_" + i;
                String chunk = chunks.get(i);
                
                // Generate embedding
                Embedding embedding = embeddingModel.embed(chunk);
                List<Double> embeddingVector = embedding.vectorAsList().stream()
                        .map(Float::doubleValue)
                        .toList();
                
                // Store chunk metadata
                DocumentChunk documentChunk = DocumentChunk.builder()
                        .documentId(documentId)
                        .chunkId(chunkId)
                        .content(chunk)
                        .page(i + 1)
                        .embedding(embeddingVector)
                        .build();
                
                // Store in Redis
                redisTemplate.opsForValue().set(VECTOR_KEY_PREFIX + chunkId, documentChunk);
                
                log.debug("Stored embedding for chunk: {}", chunkId);
            }
            
            log.info("Successfully stored embeddings for document: {}", documentId);
            
        } catch (Exception e) {
            log.error("Error storing document embeddings for document: {}", documentId, e);
            throw new RuntimeException("Failed to store document embeddings", e);
        }
    }

    public List<DocumentChunk> findSimilarChunks(String query, List<String> documentIds, int topK) {
        try {
            // Generate query embedding
            Embedding queryEmbedding = embeddingModel.embed(query);
            List<Double> queryEmbeddingVector = queryEmbedding.vectorAsList().stream()
                    .map(Float::doubleValue)
                    .toList();
            
            List<DocumentChunk> allChunks = new ArrayList<>();
            
            // Retrieve all chunks for specified documents
            for (String documentId : documentIds) {
                List<DocumentChunk> documentChunks = getDocumentChunks(documentId);
                allChunks.addAll(documentChunks);
            }
            
            // Calculate similarity scores
            List<DocumentChunk> rankedChunks = allChunks.stream()
                    .peek(chunk -> {
                        double similarity = calculateCosineSimilarity(queryEmbeddingVector, chunk.getEmbedding());
                        chunk.setRelevanceScore(similarity);
                    })
                    .sorted((a, b) -> Double.compare(b.getRelevanceScore(), a.getRelevanceScore()))
                    .limit(topK)
                    .toList();
            
            log.info("Found {} similar chunks for query: {}", rankedChunks.size(), query);
            return rankedChunks;
            
        } catch (Exception e) {
            log.error("Error finding similar chunks for query: {}", query, e);
            throw new RuntimeException("Failed to find similar chunks", e);
        }
    }

    private List<String> splitIntoChunks(String content) {
        List<String> chunks = new ArrayList<>();
        int contentLength = content.length();
        
        for (int i = 0; i < contentLength; i += CHUNK_SIZE - CHUNK_OVERLAP) {
            int endIndex = Math.min(i + CHUNK_SIZE, contentLength);
            String chunk = content.substring(i, endIndex);
            
            // Ensure we don't break words
            if (endIndex < contentLength && !Character.isWhitespace(content.charAt(endIndex))) {
                int lastSpace = chunk.lastIndexOf(' ');
                if (lastSpace > 0) {
                    chunk = chunk.substring(0, lastSpace);
                }
            }
            
            chunks.add(chunk.trim());
        }
        
        return chunks;
    }

    private List<DocumentChunk> getDocumentChunks(String documentId) {
        List<DocumentChunk> chunks = new ArrayList<>();
        
        // In a real implementation, you would query Redis or a vector database
        // For now, we'll simulate retrieving chunks
        String pattern = VECTOR_KEY_PREFIX + documentId + "_chunk_*";
        
        try {
            // This is a simplified approach - in production, use Redis SCAN or vector DB query
            for (int i = 0; i < 10; i++) { // Assume max 10 chunks per document
                String chunkKey = VECTOR_KEY_PREFIX + documentId + "_chunk_" + i;
                DocumentChunk chunk = (DocumentChunk) redisTemplate.opsForValue().get(chunkKey);
                if (chunk != null) {
                    chunks.add(chunk);
                }
            }
        } catch (Exception e) {
            log.warn("Error retrieving chunks for document: {}", documentId, e);
        }
        
        return chunks;
    }

    private double calculateCosineSimilarity(List<Double> vectorA, List<Double> vectorB) {
        if (vectorA.size() != vectorB.size()) {
            throw new IllegalArgumentException("Vectors must have the same dimension");
        }
        
        double dotProduct = IntStream.range(0, vectorA.size())
                .mapToDouble(i -> vectorA.get(i) * vectorB.get(i))
                .sum();
        
        double normA = Math.sqrt(vectorA.stream().mapToDouble(x -> x * x).sum());
        double normB = Math.sqrt(vectorB.stream().mapToDouble(x -> x * x).sum());
        
        if (normA == 0.0 || normB == 0.0) {
            return 0.0;
        }
        
        return dotProduct / (normA * normB);
    }

    @Data
    @Builder
    public static class DocumentChunk {
        private String documentId;
        private String chunkId;
        private String content;
        private int page;
        private List<Double> embedding;
        private double relevanceScore;
    }
}