package com.shibam.ai.gateway.repository;

import com.shibam.ai.gateway.entity.Document;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface DocumentRepository extends JpaRepository<Document, String> {
    
    List<Document> findByFilenameContainingIgnoreCase(String filename);
    
    List<Document> findByAnalysisType(String analysisType);
    
    List<Document> findByCreatedAtBetween(LocalDateTime startDate, LocalDateTime endDate);
    
    @Query("SELECT d FROM Document d WHERE d.confidence >= :minConfidence ORDER BY d.confidence DESC")
    List<Document> findByMinConfidence(@Param("minConfidence") double minConfidence);
    
    @Query("SELECT d FROM Document d WHERE d.sentiment = :sentiment")
    List<Document> findBySentiment(@Param("sentiment") String sentiment);
    
    Optional<Document> findByFilename(String filename);
    
    @Query("SELECT COUNT(d) FROM Document d WHERE d.createdAt >= :date")
    long countDocumentsCreatedAfter(@Param("date") LocalDateTime date);
}