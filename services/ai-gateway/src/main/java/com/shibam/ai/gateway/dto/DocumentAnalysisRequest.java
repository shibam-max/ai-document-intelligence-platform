package com.shibam.ai.gateway.dto;

import lombok.Builder;
import lombok.Data;
import org.springframework.web.multipart.MultipartFile;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

@Data
@Builder
public class DocumentAnalysisRequest {
    
    @NotNull(message = "File is required")
    private MultipartFile file;
    
    @Size(max = 50, message = "Analysis type must be less than 50 characters")
    private String analysisType;
    
    private String language;
    private boolean extractEntities;
    private boolean analyzeSentiment;
    private boolean generateSummary;
}