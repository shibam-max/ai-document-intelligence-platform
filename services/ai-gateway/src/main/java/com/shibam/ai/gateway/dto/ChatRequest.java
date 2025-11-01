package com.shibam.ai.gateway.dto;

import lombok.Builder;
import lombok.Data;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;
import java.util.List;

@Data
@Builder
public class ChatRequest {
    
    @NotBlank(message = "Question is required")
    @Size(max = 1000, message = "Question must be less than 1000 characters")
    private String question;
    
    @NotEmpty(message = "At least one document ID is required")
    private List<String> documentIds;
    
    private String model = "gpt-3.5-turbo";
    private int maxTokens = 1000;
    private double temperature = 0.7;
    private String conversationId;
}