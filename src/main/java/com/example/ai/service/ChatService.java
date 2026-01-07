package com.example.ai.service;

import dev.langchain4j.data.segment.TextSegment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 聊天服务类 - 封装所有 Spring AI 相关逻辑
 * 该服务负责处理与 AI 模型的交互，保持与 UI 层的解耦
 */
@Service
public class ChatService {

    private static final Logger log = LoggerFactory.getLogger(ChatService.class);
    
    private final ChatModel chatModel;
    private final DocumentService documentService;

    public ChatService(ChatModel chatModel, DocumentService documentService) {
        this.chatModel = chatModel;
        this.documentService = documentService;
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
     * 使用 RAG 增强：在发起 AI 请求前，先检索相关文档片段
     * 
     * @param prompt 用户输入的提示词
     * @return AI 模型的响应流（Flux<String>）
     * @throws IllegalArgumentException 如果 prompt 为空或 null
     */
    public Flux<String> streamResponse(String prompt) {
        if (prompt == null || prompt.trim().isEmpty()) {
            return Flux.error(new IllegalArgumentException("prompt 参数不能为空"));
        }

        log.info("========== 开始 RAG 增强的 AI 请求 ==========");
        log.info("用户输入: {}", prompt);
        
        try {
            // 在发起 AI 请求前，先检索相关文档片段
            log.info("开始执行文档检索...");
            List<DocumentService.SearchResult> searchResults = documentService.search(prompt);
            log.info("检索完成，找到 {} 个相关切片", searchResults.size());
            
            // 构建增强的提示词
            String augmentedPrompt = buildAugmentedPrompt(prompt, searchResults);
            log.info("已构建增强提示词，长度: {} 字符", augmentedPrompt.length());
            
            // 创建包含 System Message 和 User Message 的 Prompt
            Prompt chatPrompt = new Prompt(
                List.of(
                    new SystemMessage(buildSystemRole()),
                    new UserMessage(augmentedPrompt)
                )
            );
            log.info("已创建增强后的 Prompt，准备发送请求到 AI 模型");
            
            // 用于累积完整的响应内容
            AtomicReference<String> fullResponse = new AtomicReference<>("");
            
            return chatModel.stream(chatPrompt)
                    .doOnSubscribe(subscription -> {
                        log.info("RAG 增强请求已发送，等待响应...");
                    })
                    .map(response -> {
                        String text = response.getResult().getOutput().getText();
                        log.debug("收到流式响应片段，长度: {}", text != null ? text.length() : 0);
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
                        log.debug("流式响应 onNext 被调用，chunk 长度: {}", chunk != null ? chunk.length() : 0);
                    })
                    .doOnComplete(() -> {
                        String completeResponse = fullResponse.get();
                        log.info("========== RAG 增强的 AI 请求完成 ==========");
                        log.info("完整响应长度: {} 字符", completeResponse.length());
                    })
                    .onErrorResume(error -> {
                        log.error("========== RAG 增强的 AI 请求失败 ==========");
                        log.error("错误类型: {}", error.getClass().getName());
                        log.error("错误消息: {}", error.getMessage());
                        
                        // 特殊处理常见的 HTTP 错误
                        String userFriendlyMessage = getErrorMessage(error);
                        log.error("用户友好提示: {}", userFriendlyMessage);
                        log.error("完整错误堆栈:", error);
                        
                        return Flux.error(new RuntimeException(userFriendlyMessage, error));
                    });
        } catch (Exception e) {
            log.error("========== 创建 RAG 增强请求时发生异常 ==========");
            log.error("异常类型: {}", e.getClass().getName());
            log.error("异常消息: {}", e.getMessage());
            log.error("完整异常堆栈:", e);
            
            String userFriendlyMessage = getErrorMessage(e);
            return Flux.error(new RuntimeException(userFriendlyMessage, e));
        }
    }
    
    /**
     * 构建系统角色提示词
     * 
     * @return 系统角色提示词
     */
    private String buildSystemRole() {
        return "You are an expert on 'Effective Java'. Use the following context to answer. If not in context, say so.";
    }
    
    /**
     * 构建增强的提示词（Augmented Prompt）
     * 包含上下文（检索到的切片内容）和用户问题
     * 
     * @param originalPrompt 原始用户提示词
     * @param searchResults 检索结果列表
     * @return 增强后的提示词
     */
    private String buildAugmentedPrompt(String originalPrompt, List<DocumentService.SearchResult> searchResults) {
        StringBuilder augmentedPrompt = new StringBuilder();
        
        // 添加上下文：参考以下书中片段
        augmentedPrompt.append("Context:\n");
        if (searchResults == null || searchResults.isEmpty()) {
            augmentedPrompt.append("(No relevant context found)\n\n");
        } else {
            for (int i = 0; i < searchResults.size(); i++) {
                DocumentService.SearchResult result = searchResults.get(i);
                TextSegment segment = result.getSegment();
                augmentedPrompt.append(String.format("[Chunk %d]\n", i + 1));
                augmentedPrompt.append(segment.text());
                augmentedPrompt.append("\n\n");
            }
        }
        
        // 添加用户问题
        augmentedPrompt.append("User question: ");
        augmentedPrompt.append(originalPrompt);
        
        return augmentedPrompt.toString();
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

    /**
     * RAG 增强的流式响应
     * 使用文档检索增强生成（Retrieval-Augmented Generation）
     * 所有操作都是异步的，不会阻塞 UI 线程
     * 
     * @param userMessage 用户消息
     * @return 包含响应流和检索结果的包装对象（检索结果以 Mono 形式提供）
     */
    public RagResponse streamRagResponse(String userMessage) {
        if (userMessage == null || userMessage.trim().isEmpty()) {
            return new RagResponse(
                Flux.error(new IllegalArgumentException("userMessage 参数不能为空")),
                Mono.just(List.<DocumentService.SearchResult>of())
            );
        }

        log.info("========== 开始 RAG 增强请求 ==========");
        log.info("用户输入: {}", userMessage);

        // 异步执行查询翻译和 RAG 检索，不阻塞 UI 线程
        Mono<List<DocumentService.SearchResult>> searchResultsMono = Mono.fromCallable(() -> {
            // 步骤 1: 语言对齐 - 如果需要，将查询翻译为英文
            String searchQuery = userMessage;
            if (!isEnglish(userMessage)) {
                log.info("========== 语言对齐：检测到非英文查询 ==========");
                log.info("原始查询（用户输入）: {}", userMessage);
                log.info("开始翻译为英文搜索关键词...");
                try {
                    String englishQuery = translateToEnglishKeywords(userMessage);
                    log.info("========== 翻译完成 ==========");
                    log.info("Translated Search Query: {}", englishQuery);
                    searchQuery = englishQuery;
                } catch (Exception e) {
                    log.warn("翻译失败，使用原始查询进行搜索: {}", e.getMessage());
                    // 翻译失败时，继续使用原始查询
                }
            } else {
                log.info("查询已经是英文，跳过翻译步骤（节省 API 成本）");
            }
            
            // 步骤 2: 使用（可能翻译后的）查询执行文档检索
            log.info("开始执行文档检索，搜索查询: {}", searchQuery);
            List<DocumentService.SearchResult> results = documentService.search(searchQuery);
            log.info("检索完成，找到 {} 个相关切片", results.size());
            return results;
        }).subscribeOn(reactor.core.scheduler.Schedulers.boundedElastic())
          .doOnError(error -> {
              log.error("文档检索失败", error);
          })
          .onErrorReturn(List.of());

        // 构建增强后的 Prompt 并流式响应
        Flux<String> responseStream = searchResultsMono.flatMapMany(searchResults -> {
            // 构建 System Prompt
            String systemPrompt = buildSystemPrompt(searchResults);
            log.info("构建的 System Prompt 长度: {} 字符", systemPrompt.length());

            // 创建包含 System Message 和 User Message 的 Prompt
            Prompt enhancedPrompt = new Prompt(
                List.of(
                    new SystemMessage(systemPrompt),
                    new UserMessage(userMessage)
                )
            );

            log.info("已创建增强后的 Prompt，准备发送请求到 AI 模型");

            // 用于累积完整的响应内容
            AtomicReference<String> fullResponse = new AtomicReference<>("");

            return chatModel.stream(enhancedPrompt)
                    .doOnSubscribe(subscription -> {
                        log.info("RAG 请求已发送，等待响应...");
                    })
                    .map(response -> {
                        String text = response.getResult().getOutput().getText();
                        log.debug("收到 RAG 流式响应片段，长度: {}", text != null ? text.length() : 0);
                        if (text != null && !text.isEmpty()) {
                            String current = fullResponse.get();
                            fullResponse.set(current + text);
                        }
                        return text != null ? text : "";
                    })
                    .doOnComplete(() -> {
                        String completeResponse = fullResponse.get();
                        log.info("========== RAG 请求完成 ==========");
                        log.info("完整响应长度: {} 字符", completeResponse.length());
                    })
                    .onErrorResume(error -> {
                        log.error("========== RAG 请求失败 ==========");
                        log.error("错误类型: {}", error.getClass().getName());
                        log.error("错误消息: {}", error.getMessage());
                        
                        String userFriendlyMessage = getErrorMessage(error);
                        log.error("用户友好提示: {}", userFriendlyMessage);
                        
                        return Flux.error(new RuntimeException(userFriendlyMessage, error));
                    });
        });

        return new RagResponse(responseStream, searchResultsMono);
    }

    /**
     * 检测文本是否主要是英文
     * 简单启发式方法：如果文本中超过 50% 的字符是英文字母，则认为主要是英文
     * 
     * @param text 待检测的文本
     * @return 如果主要是英文返回 true，否则返回 false
     */
    private boolean isEnglish(String text) {
        if (text == null || text.trim().isEmpty()) {
            return true; // 空文本默认为英文
        }
        
        int totalChars = 0;
        int englishChars = 0;
        
        for (char c : text.toCharArray()) {
            if (Character.isLetter(c)) {
                totalChars++;
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                    englishChars++;
                }
            }
        }
        
        // 如果没有字母，默认为英文
        if (totalChars == 0) {
            return true;
        }
        
        // 如果英文字母占比超过 50%，认为是英文
        return (double) englishChars / totalChars > 0.5;
    }
    
    /**
     * 将中文技术问题翻译为英文搜索关键词
     * 使用 AI 模型进行翻译，确保翻译结果适合用于技术文档搜索
     * 
     * @param chineseQuery 中文查询
     * @return 英文搜索关键词
     * @throws Exception 如果翻译失败
     */
    private String translateToEnglishKeywords(String chineseQuery) throws Exception {
        String translationPrompt = String.format(
            "Translate the following Chinese technical question into English search keywords for an English technical book. " +
            "Return ONLY the translated English keywords, without any explanation or additional text: %s",
            chineseQuery
        );
        
        log.debug("翻译提示词: {}", translationPrompt);
        
        Prompt translationPromptObj = new Prompt(new UserMessage(translationPrompt));
        ChatResponse response = chatModel.call(translationPromptObj);
        
        String translatedQuery = response.getResult().getOutput().getText();
        
        if (translatedQuery == null || translatedQuery.trim().isEmpty()) {
            throw new RuntimeException("翻译结果为空");
        }
        
        // 清理翻译结果（移除可能的引号、多余空格等）
        translatedQuery = translatedQuery.trim();
        if (translatedQuery.startsWith("\"") && translatedQuery.endsWith("\"")) {
            translatedQuery = translatedQuery.substring(1, translatedQuery.length() - 1);
        }
        if (translatedQuery.startsWith("'") && translatedQuery.endsWith("'")) {
            translatedQuery = translatedQuery.substring(1, translatedQuery.length() - 1);
        }
        
        return translatedQuery.trim();
    }
    
    /**
     * 构建 System Prompt，包含检索到的参考资料
     * 
     * @param searchResults 检索结果列表
     * @return 构建的 System Prompt
     */
    private String buildSystemPrompt(List<DocumentService.SearchResult> searchResults) {
        StringBuilder promptBuilder = new StringBuilder();
        promptBuilder.append("你是一个基于《Effective Java》的专业助手。");
        promptBuilder.append("请根据以下提供的【参考资料】回答用户问题。");
        promptBuilder.append("如果资料中没有相关信息，请诚实说明。");
        promptBuilder.append("\n\n参考资料如下：\n\n");

        if (searchResults == null || searchResults.isEmpty()) {
            promptBuilder.append("（未找到相关参考资料）");
        } else {
            for (int i = 0; i < searchResults.size(); i++) {
                DocumentService.SearchResult result = searchResults.get(i);
                TextSegment segment = result.getSegment();
                promptBuilder.append(String.format("[切片 %d]\n", i + 1));
                promptBuilder.append(segment.text());
                promptBuilder.append("\n\n");
            }
        }

        return promptBuilder.toString();
    }

    /**
     * RAG 响应包装类，包含响应流和检索结果（异步）
     */
    public static class RagResponse {
        private final Flux<String> responseStream;
        private final Mono<List<DocumentService.SearchResult>> searchResultsMono;

        public RagResponse(Flux<String> responseStream, Mono<List<DocumentService.SearchResult>> searchResultsMono) {
            this.responseStream = responseStream;
            this.searchResultsMono = searchResultsMono;
        }

        public Flux<String> getResponseStream() {
            return responseStream;
        }

        public Mono<List<DocumentService.SearchResult>> getSearchResultsMono() {
            return searchResultsMono;
        }
    }
}

