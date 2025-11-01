package com.shibam.ai.gateway.dto;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Builder
public class ChatResponse {
    
    private String answer;
    private List<SourceReference> sources;
    private double confidence;
    private String model;
    private LocalDateTime timestamp;
    private String conversationId;
    private int tokensUsed;
}