package com.example.ai.service;

import dev.langchain4j.data.segment.TextSegment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 聊天服务类 - 封装所有 Spring AI 相关逻辑
 * 该服务负责处理与 AI 模型的交互，保持与 UI 层的解耦
 */
@Service
public class ChatService {

    private static final Logger log = LoggerFactory.getLogger(ChatService.class);
    private static final int MEMORY_CAPACITY = 10;
    
    private final ChatModel chatModel;
    private final DocumentService documentService;
    private final List<Message> chatMemory;
    private final Object memoryLock = new Object();

    public ChatService(ChatModel chatModel, DocumentService documentService) {
        this.chatModel = chatModel;
        this.documentService = documentService;
        // 初始化对话记忆，使用线程安全的 LinkedList，容量设为 10 条消息
        this.chatMemory = Collections.synchronizedList(new LinkedList<>());
        log.info("已初始化对话记忆，容量: {} 条消息", MEMORY_CAPACITY);
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
            
            // 步骤 2: 生成 HyDE (Hypothetical Document Embeddings) 假设答案
            String hydeQuery = searchQuery;
            try {
                log.info("========== 开始生成 HyDE 假设答案 ==========");
                log.info("原始查询: {}", searchQuery);
                String hypotheticalAnswer = generateHyDEAnswer(searchQuery);
                log.info("========== HyDE 假设答案生成完成 ==========");
                log.info("HyDE Hypothetical Answer: {}", hypotheticalAnswer);
                hydeQuery = hypotheticalAnswer;
            } catch (Exception e) {
                log.warn("HyDE 生成失败，使用原始查询进行搜索: {}", e.getMessage());
                // HyDE 生成失败时，继续使用原始查询
            }
            
            // 步骤 3: 使用 HyDE 假设答案执行文档检索（检索更多结果用于重排序）
            int initialRetrievalSize = 15; // 初始检索 15 个 chunks（使用 HyDE 后可以减少检索数量）
            log.info("开始执行文档检索，HyDE 查询: {}，初始检索数量: {}", hydeQuery, initialRetrievalSize);
            List<DocumentService.SearchResult> initialResults = documentService.search(hydeQuery, initialRetrievalSize);
            log.info("初始检索完成，找到 {} 个相关切片", initialResults.size());
            
            // 步骤 4: 重排序 - 使用本地评分模型对结果进行重新排序
            log.info("开始执行重排序，从 {} 个结果中选择 TOP 5", initialResults.size());
            List<DocumentService.SearchResult> rerankedResults = rerankResults(initialResults, userMessage, 5);
            log.info("重排序完成，最终选择 {} 个最相关的切片", rerankedResults.size());
            
            return rerankedResults;
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

            // 获取对话历史（线程安全）
            List<Message> conversationHistory;
            synchronized (memoryLock) {
                conversationHistory = new ArrayList<>(chatMemory);
            }
            log.info("对话历史消息数: {}", conversationHistory.size());

            // 构建完整的消息列表：System Message + 对话历史 + 当前用户消息
            List<Message> messages = new ArrayList<>();
            messages.add(new SystemMessage(systemPrompt));
            
            // 添加对话历史（排除 SystemMessage，只保留 UserMessage 和 AssistantMessage）
            for (Message historyMessage : conversationHistory) {
                if (!(historyMessage instanceof SystemMessage)) {
                    messages.add(historyMessage);
                }
            }
            
            // 添加当前用户消息
            messages.add(new UserMessage(userMessage));

            // 创建包含对话历史的 Prompt
            Prompt enhancedPrompt = new Prompt(messages);

            log.info("已创建增强后的 Prompt（包含 {} 条历史消息），准备发送请求到 AI 模型", conversationHistory.size());

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
                        
                        // 将用户问题和 AI 回答存入记忆（线程安全）
                        try {
                            synchronized (memoryLock) {
                                chatMemory.add(new UserMessage(userMessage));
                                chatMemory.add(new AssistantMessage(completeResponse));
                                
                                // 如果超过容量，移除最旧的消息（保持容量为 10 条）
                                while (chatMemory.size() > MEMORY_CAPACITY) {
                                    chatMemory.remove(0);
                                }
                            }
                            log.info("已将用户问题和 AI 回答存入对话记忆，当前记忆大小: {}", chatMemory.size());
                        } catch (Exception e) {
                            log.warn("保存对话记忆失败: {}", e.getMessage());
                        }
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
     * 生成 HyDE (Hypothetical Document Embeddings) 假设答案
     * 通过生成一个假设的答案来改善检索准确性，使查询更接近文档的语义空间
     * 
     * @param query 用户查询（已翻译为英文）
     * @return 假设答案文本
     * @throws Exception 如果生成失败
     */
    private String generateHyDEAnswer(String query) throws Exception {
        String hydePrompt = String.format(
            "Please write a brief, technical answer to the following question as if it were an excerpt from a professional Java book like 'Effective Java'. " +
            "The answer should be concise (2-3 sentences), technical, and written in the style of a programming book. " +
            "Do not include the question itself, only provide the answer. " +
            "Question: %s",
            query
        );
        
        log.debug("HyDE 提示词: {}", hydePrompt);
        
        Prompt hydePromptObj = new Prompt(new UserMessage(hydePrompt));
        ChatResponse response = chatModel.call(hydePromptObj);
        
        String hypotheticalAnswer = response.getResult().getOutput().getText();
        
        if (hypotheticalAnswer == null || hypotheticalAnswer.trim().isEmpty()) {
            throw new RuntimeException("HyDE 假设答案生成结果为空");
        }
        
        // 清理结果（移除可能的引号、多余空格等）
        hypotheticalAnswer = hypotheticalAnswer.trim();
        if (hypotheticalAnswer.startsWith("\"") && hypotheticalAnswer.endsWith("\"")) {
            hypotheticalAnswer = hypotheticalAnswer.substring(1, hypotheticalAnswer.length() - 1);
        }
        if (hypotheticalAnswer.startsWith("'") && hypotheticalAnswer.endsWith("'")) {
            hypotheticalAnswer = hypotheticalAnswer.substring(1, hypotheticalAnswer.length() - 1);
        }
        
        return hypotheticalAnswer.trim();
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
     * 重排序搜索结果
     * 使用本地评分模型对检索结果进行重新排序，选择最相关的 TOP K 个结果
     * 
     * @param initialResults 初始检索结果
     * @param userQuery 用户原始查询（用于评分）
     * @param topK 最终返回的结果数量
     * @return 重排序后的 TOP K 个结果
     */
    private List<DocumentService.SearchResult> rerankResults(
            List<DocumentService.SearchResult> initialResults, 
            String userQuery, 
            int topK) {
        
        if (initialResults == null || initialResults.isEmpty()) {
            return List.of();
        }
        
        if (initialResults.size() <= topK) {
            // 如果初始结果数量已经小于等于 topK，直接返回
            log.info("初始结果数量 ({}) 已小于等于目标数量 ({})，跳过重排序", initialResults.size(), topK);
            return initialResults;
        }
        
        log.info("开始重排序 {} 个结果，目标选择 TOP {}", initialResults.size(), topK);
        
        // 使用本地评分模型对每个结果进行评分
        List<ScoredResult> scoredResults = initialResults.stream()
                .map(result -> {
                    double rerankScore = calculateRerankScore(result, userQuery);
                    return new ScoredResult(result, rerankScore);
                })
                .sorted((a, b) -> Double.compare(b.rerankScore, a.rerankScore)) // 降序排序
                .limit(topK)
                .collect(java.util.stream.Collectors.toList());
        
        if (!scoredResults.isEmpty()) {
            double minScore = scoredResults.get(scoredResults.size() - 1).rerankScore;
            double maxScore = scoredResults.get(0).rerankScore;
            log.info("重排序完成，TOP {} 个结果的得分范围: {:.4f} - {:.4f}", 
                    topK, String.format("%.4f", minScore), String.format("%.4f", maxScore));
        } else {
            log.info("重排序完成，但未找到任何结果");
        }
        
        return scoredResults.stream()
                .map(sr -> sr.result)
                .collect(java.util.stream.Collectors.toList());
    }
    
    /**
     * 计算重排序得分
     * 结合向量相似度得分和文本相关性得分
     * 
     * @param result 搜索结果
     * @param userQuery 用户查询
     * @return 重排序得分（0.0 - 1.0）
     */
    private double calculateRerankScore(DocumentService.SearchResult result, String userQuery) {
        String text = result.getSegment().text().toLowerCase();
        String query = userQuery.toLowerCase();
        
        // 1. 向量相似度得分（归一化到 0-1，权重 0.4）
        double vectorScore = Math.max(0.0, Math.min(1.0, result.getScore()));
        double weightedVectorScore = vectorScore * 0.4;
        
        // 2. 关键词匹配得分（权重 0.3）
        double keywordScore = calculateKeywordMatchScore(text, query);
        double weightedKeywordScore = keywordScore * 0.3;
        
        // 3. 文本长度得分（偏好中等长度的文本，权重 0.1）
        double lengthScore = calculateLengthScore(text);
        double weightedLengthScore = lengthScore * 0.1;
        
        // 4. 查询词密度得分（查询词在文本中的出现频率，权重 0.2）
        double densityScore = calculateQueryDensityScore(text, query);
        double weightedDensityScore = densityScore * 0.2;
        
        // 综合得分
        double finalScore = weightedVectorScore + weightedKeywordScore + weightedLengthScore + weightedDensityScore;
        
        return Math.max(0.0, Math.min(1.0, finalScore));
    }
    
    /**
     * 计算关键词匹配得分
     * 检查查询中的关键词是否出现在文本中
     */
    private double calculateKeywordMatchScore(String text, String query) {
        if (text == null || query == null || text.isEmpty() || query.isEmpty()) {
            return 0.0;
        }
        
        // 提取查询中的关键词（去除常见停用词）
        String[] stopWords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "can", "this", "that", "these", "those", "what", "which", "who", "where", "when", "why", "how"};
        java.util.Set<String> stopWordSet = java.util.Set.of(stopWords);
        
        String[] queryWords = query.split("\\s+");
        int matchedWords = 0;
        int totalWords = 0;
        
        for (String word : queryWords) {
            word = word.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
            if (!word.isEmpty() && !stopWordSet.contains(word)) {
                totalWords++;
                if (text.contains(word)) {
                    matchedWords++;
                }
            }
        }
        
        return totalWords > 0 ? (double) matchedWords / totalWords : 0.0;
    }
    
    /**
     * 计算文本长度得分
     * 偏好中等长度的文本（100-500 字符）
     */
    private double calculateLengthScore(String text) {
        if (text == null) {
            return 0.0;
        }
        
        int length = text.length();
        int optimalMin = 100;
        int optimalMax = 500;
        
        if (length < optimalMin) {
            // 太短的文本得分较低
            return (double) length / optimalMin * 0.5;
        } else if (length <= optimalMax) {
            // 中等长度的文本得分最高
            return 1.0;
        } else {
            // 太长的文本得分逐渐降低
            double excess = length - optimalMax;
            double penalty = Math.min(0.5, excess / optimalMax);
            return 1.0 - penalty;
        }
    }
    
    /**
     * 计算查询词密度得分
     * 查询词在文本中的出现频率
     */
    private double calculateQueryDensityScore(String text, String query) {
        if (text == null || query == null || text.isEmpty() || query.isEmpty()) {
            return 0.0;
        }
        
        String[] queryWords = query.toLowerCase().split("\\s+");
        int totalOccurrences = 0;
        int totalWords = 0;
        
        for (String word : queryWords) {
            word = word.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
            if (!word.isEmpty()) {
                totalWords++;
                // 计算单词在文本中的出现次数
                int count = (text.length() - text.replace(word, "").length()) / word.length();
                totalOccurrences += count;
            }
        }
        
        if (totalWords == 0 || text.length() == 0) {
            return 0.0;
        }
        
        // 归一化：出现次数 / (文本长度 / 平均单词长度)
        double avgWordLength = 5.0; // 假设平均单词长度为 5
        double normalizedDensity = (double) totalOccurrences / (text.length() / avgWordLength);
        
        // 使用 sigmoid 函数将密度映射到 0-1
        return Math.min(1.0, normalizedDensity / 2.0);
    }
    
    /**
     * 带重排序得分的搜索结果包装类
     */
    private static class ScoredResult {
        final DocumentService.SearchResult result;
        final double rerankScore;
        
        ScoredResult(DocumentService.SearchResult result, double rerankScore) {
            this.result = result;
            this.rerankScore = rerankScore;
        }
    }
    
    /**
     * 清理对话记忆
     * 在切换 Tab（从通用聊天到书本助手）时调用
     */
    public void clearMemory() {
        synchronized (memoryLock) {
            chatMemory.clear();
        }
        log.info("已清理对话记忆");
    }
    
    /**
     * 获取当前对话记忆的消息数量
     * 
     * @return 消息数量
     */
    public int getMemorySize() {
        synchronized (memoryLock) {
            return chatMemory.size();
        }
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

