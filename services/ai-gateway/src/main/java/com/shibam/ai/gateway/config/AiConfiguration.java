package com.shibam.ai.gateway.config;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.GenericJackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.StringRedisSerializer;

@Configuration
public class AiConfiguration {

    @Value("${openai.api.key:demo-key}")
    private String openAiApiKey;

    @Bean
    public ChatLanguageModel chatLanguageModel() {
        if ("demo-key".equals(openAiApiKey)) {
            // Return a mock implementation for demo
            return new MockChatLanguageModel();
        }
        return OpenAiChatModel.builder()
                .apiKey(openAiApiKey)
                .modelName("gpt-3.5-turbo")
                .temperature(0.7)
                .maxTokens(1000)
                .build();
    }

    @Bean
    public EmbeddingModel embeddingModel() {
        return new AllMiniLmL6V2EmbeddingModel();
    }

    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory connectionFactory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(connectionFactory);
        
        template.setKeySerializer(new StringRedisSerializer());
        template.setHashKeySerializer(new StringRedisSerializer());
        template.setValueSerializer(new GenericJackson2JsonRedisSerializer());
        template.setHashValueSerializer(new GenericJackson2JsonRedisSerializer());
        
        template.afterPropertiesSet();
        return template;
    }

    // Mock implementation for demo purposes
    private static class MockChatLanguageModel implements ChatLanguageModel {
        @Override
        public String generate(String userMessage) {
            return "Mock AI Response: This document contains important information. " +
                   "Summary: The document discusses key topics with relevant details. " +
                   "Entities: Company, Date, Amount. Sentiment: Neutral. Confidence: 0.85";
        }
    }
}