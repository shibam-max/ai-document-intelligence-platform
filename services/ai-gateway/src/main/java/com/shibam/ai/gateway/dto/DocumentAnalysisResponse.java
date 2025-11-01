package com.shibam.ai.gateway.dto;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Builder
public class DocumentAnalysisResponse {
    
    private String documentId;
    private String summary;
    private List<String> keyEntities;
    private String sentiment;
    private double confidence;
    private String analysisType;
    private LocalDateTime analyzedAt;
    private String filename;
    private long fileSize;
    private String contentType;
}