package com.example.ai.config;

import org.springframework.context.annotation.Configuration;

/**
 * OpenRouter 配置类
 * 用于配置 OpenRouter API 的额外请求头（可选，但推荐）
 */
@Configuration
public class OpenRouterConfig {

    /**
     * 自定义 OpenAiApi，添加 OpenRouter 推荐的请求头
     * 
     * 注意：如果 Spring AI 自动配置的 OpenAiApi 已经工作，这个配置可能不需要
     * 但如果需要添加额外的请求头（如 HTTP-Referer），可以通过这个配置类实现
     */
    // 如果需要添加额外的请求头，可以取消下面的注释并实现
    /*
    @Bean
    public OpenAiApi openAiApi(
            @Value("${spring.ai.openai.api-key}") String apiKey,
            @Value("${spring.ai.openai.base-url:https://api.openai.com}") String baseUrl) {
        
        WebClient.Builder webClientBuilder = WebClient.builder()
                .baseUrl(baseUrl)
                .defaultHeader("Authorization", "Bearer " + apiKey)
                // OpenRouter 推荐的请求头（可选）
                .defaultHeader("HTTP-Referer", "http://localhost:8080") // 您的应用 URL
                .defaultHeader("X-Title", "Java AI Integration Lab"); // 您的应用名称
        
        return new OpenAiApi(baseUrl, apiKey, webClientBuilder);
    }
    */
}
