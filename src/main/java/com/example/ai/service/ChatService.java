package com.example.ai.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Flux;

import java.util.concurrent.atomic.AtomicReference;

/**
 * 聊天服务类 - 封装所有 Spring AI 相关逻辑
 * 该服务负责处理与 AI 模型的交互，保持与 UI 层的解耦
 */
@Service
public class ChatService {

    private static final Logger log = LoggerFactory.getLogger(ChatService.class);
    
    private final ChatModel chatModel;

    public ChatService(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    /**
     * 同步调用 AI 模型，返回完整响应
     * 
     * @param prompt 用户输入的提示词
     * @return AI 模型的完整响应
     * @throws IllegalArgumentException 如果 prompt 为空或 null
     */
    public String getResponse(String prompt) {
        if (prompt == null || prompt.trim().isEmpty()) {
            throw new IllegalArgumentException("prompt 参数不能为空");
        }

        log.info("========== 开始 AI 请求（同步） ==========");
        log.info("用户输入: {}", prompt);
        
        try {
            Prompt chatPrompt = new Prompt(new UserMessage(prompt));
            log.info("已创建 Prompt，准备发送请求到 AI 模型");
            
            ChatResponse response = chatModel.call(chatPrompt);
            String result = response.getResult().getOutput().getText();
            
            log.info("收到完整响应，长度: {}", result != null ? result.length() : 0);
            log.info("完整响应内容: {}", result);
            log.info("========== AI 请求完成（同步） ==========");
            
            return result;
        } catch (Exception e) {
            log.error("========== AI 请求失败（同步） ==========");
            log.error("错误类型: {}", e.getClass().getName());
            log.error("错误消息: {}", e.getMessage());
            
            String userFriendlyMessage = getErrorMessage(e);
            log.error("用户友好提示: {}", userFriendlyMessage);
            log.error("完整错误堆栈:", e);
            
            throw new RuntimeException(userFriendlyMessage, e);
        }
    }

    /**
     * 流式调用 AI 模型，返回响应流
     * 
     * @param prompt 用户输入的提示词
     * @return AI 模型的响应流（Flux<String>）
     * @throws IllegalArgumentException 如果 prompt 为空或 null
     */
    public Flux<String> streamResponse(String prompt) {
        if (prompt == null || prompt.trim().isEmpty()) {
            return Flux.error(new IllegalArgumentException("prompt 参数不能为空"));
        }

        log.info("========== 开始 AI 请求 ==========");
        log.info("用户输入: {}", prompt);
        
        try {
            Prompt chatPrompt = new Prompt(new UserMessage(prompt));
            log.info("已创建 Prompt，准备发送请求到 AI 模型");
            
            // 用于累积完整的响应内容
            AtomicReference<String> fullResponse = new AtomicReference<>("");
            
            return chatModel.stream(chatPrompt)
                    .doOnSubscribe(subscription -> {
                        log.info("请求已发送，等待响应...");
                    })
                    .map(response -> {
                        String text = response.getResult().getOutput().getText();
                        log.info("收到流式响应片段，长度: {}", text != null ? text.length() : 0);
                        if (text != null && !text.isEmpty()) {
                            // 累积响应内容
                            String current = fullResponse.get();
                            fullResponse.set(current + text);
                            log.debug("片段内容: {}", text.length() > 100 ? text.substring(0, 100) + "..." : text);
                        } else {
                            log.warn("收到空的响应片段");
                        }
                        return text != null ? text : "";
                    })
                    .doOnNext(chunk -> {
                        log.info("流式响应 onNext 被调用，chunk 长度: {}", chunk != null ? chunk.length() : 0);
                    })
                    .doOnComplete(() -> {
                        String completeResponse = fullResponse.get();
                        log.info("========== AI 请求完成 ==========");
                        log.info("完整响应内容: {}", completeResponse);
                        log.info("响应长度: {} 字符", completeResponse.length());
                    })
                    .onErrorResume(error -> {
                        log.error("========== AI 请求失败 ==========");
                        log.error("错误类型: {}", error.getClass().getName());
                        log.error("错误消息: {}", error.getMessage());
                        
                        // 特殊处理常见的 HTTP 错误
                        String userFriendlyMessage = getErrorMessage(error);
                        log.error("用户友好提示: {}", userFriendlyMessage);
                        log.error("完整错误堆栈:", error);
                        
                        return Flux.error(new RuntimeException(userFriendlyMessage, error));
                    });
        } catch (Exception e) {
            log.error("========== 创建请求时发生异常 ==========");
            log.error("异常类型: {}", e.getClass().getName());
            log.error("异常消息: {}", e.getMessage());
            log.error("完整异常堆栈:", e);
            
            String userFriendlyMessage = getErrorMessage(e);
            return Flux.error(new RuntimeException(userFriendlyMessage, e));
        }
    }
    
    /**
     * 根据错误类型返回用户友好的错误消息
     */
    private String getErrorMessage(Throwable error) {
        if (error instanceof WebClientResponseException) {
            WebClientResponseException webClientException = (WebClientResponseException) error;
            int statusCode = webClientException.getStatusCode().value();
            
            switch (statusCode) {
                case 401:
                    return "❌ API 认证失败 (401 Unauthorized)\n" +
                           "请检查您的 API key 是否正确配置在 application-secrets.yaml 文件中";
                case 429:
                    return "❌ 请求频率过高 (429 Too Many Requests)\n" +
                           "OpenAI API 速率限制已触发，请稍候再试。\n" +
                           "可能原因：\n" +
                           "1. 短时间内发送了太多请求\n" +
                           "2. API 配额已用完\n" +
                           "3. 账户余额不足\n" +
                           "建议：等待几分钟后重试，或检查 OpenAI 账户状态";
                case 500:
                case 502:
                case 503:
                    return "❌ OpenAI 服务器错误 (" + statusCode + ")\n" +
                           "OpenAI 服务暂时不可用，请稍候再试";
                default:
                    return "❌ API 请求失败 (" + statusCode + "): " + error.getMessage();
            }
        }
        
        // 检查是否是 429 错误的包装异常
        Throwable cause = error.getCause();
        if (cause instanceof WebClientResponseException) {
            return getErrorMessage(cause);
        }
        
        return "❌ 处理请求时发生异常: " + error.getMessage();
    }
}

