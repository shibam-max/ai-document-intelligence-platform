package com.shibam.ai.gateway.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class SourceReference {
    
    private String documentId;
    private int page;
    private double relevanceScore;
    private String excerpt;
    private String filename;
}