package com.example.ai.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.segment.TextSegment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * RAG 评估服务
 * 提供自动化测试集生成、批量测试和 LLM-as-a-Judge 评分功能
 */
@Service
public class RagEvaluationService {

    private static final Logger log = LoggerFactory.getLogger(RagEvaluationService.class);
    
    private final ChatModel chatModel;
    private final ChatService chatService;
    private final DocumentService documentService;
    private final ObjectMapper objectMapper;
    
    // 测试集文件路径
    private static final String TEST_SET_FILE = "external-docs/test-set.json";
    private static final String REPORT_FILE = "external-docs/evaluation_report.md";
    
    // Judge AI 的 System Prompt
    private static final String JUDGE_SYSTEM_PROMPT = """
        You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
        Your task is to objectively score answers based on two criteria:
        
        1. **Faithfulness** (0-1): Does the answer accurately reflect the provided context?
           - 1.0: Answer is completely faithful to the context, no hallucinations
           - 0.5: Answer is partially faithful but contains some inaccuracies
           - 0.0: Answer contradicts or ignores the context
        
        2. **Relevance** (0-1): Does the answer address the question?
           - 1.0: Answer directly and completely addresses the question
           - 0.5: Answer partially addresses the question
           - 0.0: Answer does not address the question
        
        You must respond ONLY with a JSON object in this exact format:
        {
          "faithfulness": 0.85,
          "relevance": 0.90,
          "reasoning": "Brief explanation of scores"
        }
        
        Do not include any other text, only the JSON object.
        """;
    
    public RagEvaluationService(ChatModel chatModel, ChatService chatService, DocumentService documentService) {
        this.chatModel = chatModel;
        this.chatService = chatService;
        this.documentService = documentService;
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * 数据模型：测试问题
     */
    public static class TestQuestion {
        @JsonProperty("question")
        private String question;
        
        @JsonProperty("ground_truth")
        private String groundTruth;
        
        @JsonProperty("source_segment")
        private String sourceSegment;
        
        @JsonProperty("segment_id")
        private String segmentId;
        
        public TestQuestion() {}
        
        public TestQuestion(String question, String groundTruth, String sourceSegment, String segmentId) {
            this.question = question;
            this.groundTruth = groundTruth;
            this.sourceSegment = sourceSegment;
            this.segmentId = segmentId;
        }
        
        // Getters and Setters
        public String getQuestion() { return question; }
        public void setQuestion(String question) { this.question = question; }
        public String getGroundTruth() { return groundTruth; }
        public void setGroundTruth(String groundTruth) { this.groundTruth = groundTruth; }
        public String getSourceSegment() { return sourceSegment; }
        public void setSourceSegment(String sourceSegment) { this.sourceSegment = sourceSegment; }
        public String getSegmentId() { return segmentId; }
        public void setSegmentId(String segmentId) { this.segmentId = segmentId; }
    }
    
    /**
     * 数据模型：评估结果
     */
    public static class EvaluationResult {
        @JsonProperty("question")
        private String question;
        
        @JsonProperty("rag_answer")
        private String ragAnswer;
        
        @JsonProperty("ground_truth")
        private String groundTruth;
        
        @JsonProperty("faithfulness_score")
        private double faithfulnessScore;
        
        @JsonProperty("relevance_score")
        private double relevanceScore;
        
        @JsonProperty("judge_reasoning")
        private String judgeReasoning;
        
        @JsonProperty("sources_used")
        private List<String> sourcesUsed;
        
        public EvaluationResult() {}
        
        // Getters and Setters
        public String getQuestion() { return question; }
        public void setQuestion(String question) { this.question = question; }
        public String getRagAnswer() { return ragAnswer; }
        public void setRagAnswer(String ragAnswer) { this.ragAnswer = ragAnswer; }
        public String getGroundTruth() { return groundTruth; }
        public void setGroundTruth(String groundTruth) { this.groundTruth = groundTruth; }
        public double getFaithfulnessScore() { return faithfulnessScore; }
        public void setFaithfulnessScore(double faithfulnessScore) { this.faithfulnessScore = faithfulnessScore; }
        public double getRelevanceScore() { return relevanceScore; }
        public void setRelevanceScore(double relevanceScore) { this.relevanceScore = relevanceScore; }
        public String getJudgeReasoning() { return judgeReasoning; }
        public void setJudgeReasoning(String judgeReasoning) { this.judgeReasoning = judgeReasoning; }
        public List<String> getSourcesUsed() { return sourcesUsed; }
        public void setSourcesUsed(List<String> sourcesUsed) { this.sourcesUsed = sourcesUsed; }
    }
    
    /**
     * 数据模型：Judge 评分响应
     */
    private static class JudgeScore {
        @JsonProperty("faithfulness")
        private double faithfulness;
        
        @JsonProperty("relevance")
        private double relevance;
        
        @JsonProperty("reasoning")
        private String reasoning;
        
        public double getFaithfulness() { return faithfulness; }
        public void setFaithfulness(double faithfulness) { this.faithfulness = faithfulness; }
        public double getRelevance() { return relevance; }
        public void setRelevance(double relevance) { this.relevance = relevance; }
        public String getReasoning() { return reasoning; }
        public void setReasoning(String reasoning) { this.reasoning = reasoning; }
    }
    
    /**
     * Feature 1: Test Set Generator
     * 遍历文档片段，使用 LLM 生成 (Question, Ground Truth) 对
     * 
     * @param numQuestions 要生成的问题数量（如果为 -1，则为每个片段生成一个问题）
     * @return 生成的测试问题列表
     */
    public List<TestQuestion> generateTestSet(int numQuestions) {
        log.info("========== 开始生成测试集 ==========");
        
        // 获取所有文档片段
        List<TextSegment> segments = documentService.getChunks();
        if (segments == null || segments.isEmpty()) {
            log.warn("文档片段为空，无法生成测试集");
            return Collections.emptyList();
        }
        
        log.info("找到 {} 个文档片段", segments.size());
        
        // 确定要处理的片段数量
        int segmentsToProcess = (numQuestions == -1) ? segments.size() : Math.min(numQuestions, segments.size());
        log.info("将处理 {} 个片段生成测试问题", segmentsToProcess);
        
        // 使用并行流处理片段
        List<TestQuestion> testQuestions = segments.stream()
                .limit(segmentsToProcess)
                .parallel()
                .map(this::generateQuestionFromSegment)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        
        log.info("成功生成 {} 个测试问题", testQuestions.size());
        
        // 保存到文件
        saveTestSetToFile(testQuestions);
        
        return testQuestions;
    }
    
    /**
     * 从单个片段生成测试问题
     */
    private TestQuestion generateQuestionFromSegment(TextSegment segment) {
        try {
            String segmentText = segment.text();
            if (segmentText == null || segmentText.trim().length() < 50) {
                log.debug("片段太短，跳过: {}", segmentText != null ? segmentText.length() : 0);
                return null;
            }
            
            // 限制片段长度（避免提示词过长）
            String truncatedSegment = segmentText.length() > 1000 
                    ? segmentText.substring(0, 1000) + "..." 
                    : segmentText;
            
            String prompt = String.format(
                "Given the following excerpt from 'Effective Java', generate a test question and its ground truth answer.\n\n" +
                "Excerpt:\n%s\n\n" +
                "Please generate:\n" +
                "1. A clear, specific question that can be answered using this excerpt.\n" +
                "2. A concise ground truth answer (2-3 sentences) based on the excerpt.\n\n" +
                "Respond ONLY with a JSON object in this exact format:\n" +
                "{\n" +
                "  \"question\": \"Your question here\",\n" +
                "  \"ground_truth\": \"The answer based on the excerpt\"\n" +
                "}\n\n" +
                "Do not include any other text, only the JSON object.",
                truncatedSegment
            );
            
            Prompt questionPrompt = new Prompt(new UserMessage(prompt));
            ChatResponse response = chatModel.call(questionPrompt);
            String responseText = response.getResult().getOutput().getText();
            
            // 解析 JSON 响应
            Map<String, String> parsed = parseJsonResponse(responseText);
            if (parsed == null || !parsed.containsKey("question") || !parsed.containsKey("ground_truth")) {
                log.warn("无法解析问题生成响应: {}", responseText);
                return null;
            }
            
            String segmentId = segment.metadata() != null && segment.metadata().get("item_id") != null
                    ? segment.metadata().get("item_id")
                    : UUID.randomUUID().toString();
            
            TestQuestion testQuestion = new TestQuestion(
                    parsed.get("question"),
                    parsed.get("ground_truth"),
                    truncatedSegment,
                    segmentId
            );
            
            log.debug("成功生成测试问题: {}", parsed.get("question"));
            return testQuestion;
            
        } catch (Exception e) {
            log.error("生成测试问题时出错", e);
            return null;
        }
    }
    
    /**
     * 保存测试集到文件
     */
    private void saveTestSetToFile(List<TestQuestion> testQuestions) {
        try {
            Path filePath = Paths.get(TEST_SET_FILE);
            Files.createDirectories(filePath.getParent());
            
            objectMapper.writerWithDefaultPrettyPrinter()
                    .writeValue(filePath.toFile(), testQuestions);
            
            log.info("测试集已保存到: {}", TEST_SET_FILE);
        } catch (IOException e) {
            log.error("保存测试集失败", e);
            throw new RuntimeException("保存测试集失败", e);
        }
    }
    
    /**
     * 从文件加载测试集
     */
    public List<TestQuestion> loadTestSetFromFile() {
        try {
            File file = new File(TEST_SET_FILE);
            if (!file.exists()) {
                log.warn("测试集文件不存在: {}", TEST_SET_FILE);
                return Collections.emptyList();
            }
            
            TestQuestion[] questions = objectMapper.readValue(file, TestQuestion[].class);
            log.info("从文件加载了 {} 个测试问题", questions.length);
            return Arrays.asList(questions);
            
        } catch (IOException e) {
            log.error("加载测试集失败", e);
            return Collections.emptyList();
        }
    }
    
    /**
     * Feature 2: Batch Tester
     * 读取测试集，运行每个问题通过 RAG 管道，收集结果
     * 
     * @param testQuestions 测试问题列表（如果为 null，则从文件加载）
     * @return 评估结果列表
     */
    public List<EvaluationResult> runBatchTest(List<TestQuestion> testQuestions) {
        log.info("========== 开始批量测试 ==========");
        
        final List<TestQuestion> finalTestQuestions;
        if (testQuestions == null || testQuestions.isEmpty()) {
            finalTestQuestions = loadTestSetFromFile();
            if (finalTestQuestions.isEmpty()) {
                log.warn("测试集为空，无法运行批量测试");
                return Collections.emptyList();
            }
        } else {
            finalTestQuestions = testQuestions;
        }
        
        log.info("将测试 {} 个问题", finalTestQuestions.size());
        
        // 使用 Project Reactor 进行并行处理
        AtomicInteger completed = new AtomicInteger(0);
        
        List<Mono<EvaluationResult>> resultMonos = finalTestQuestions.stream()
                .map(testQuestion -> Mono.fromCallable(() -> {
                    try {
                        return testSingleQuestion(testQuestion);
                    } catch (Exception e) {
                        log.error("测试问题失败: {}", testQuestion.getQuestion(), e);
                        return createErrorResult(testQuestion, e.getMessage());
                    }
                })
                .subscribeOn(Schedulers.boundedElastic())
                .doOnSuccess(result -> {
                    int count = completed.incrementAndGet();
                    if (count % 10 == 0 || count == finalTestQuestions.size()) {
                        log.info("测试进度: {}/{} ({:.1f}%)", 
                                count, finalTestQuestions.size(), 
                                (double) count / finalTestQuestions.size() * 100);
                    }
                }))
                .collect(Collectors.toList());
        
        // 等待所有测试完成
        List<EvaluationResult> results = Flux.fromIterable(resultMonos)
                .flatMap(mono -> mono)
                .collectList()
                .block();
        
        log.info("批量测试完成，共测试 {} 个问题", results != null ? results.size() : 0);
        
        return results != null ? results : Collections.emptyList();
    }
    
    /**
     * 测试单个问题
     */
    private EvaluationResult testSingleQuestion(TestQuestion testQuestion) {
        log.debug("测试问题: {}", testQuestion.getQuestion());
        
        // 1. 通过 RAG 管道获取答案
        ChatService.RagResponse ragResponse = chatService.streamRagResponse(testQuestion.getQuestion());
        
        // 收集流式响应
        String ragAnswer = ragResponse.getResponseStream()
                .collectList()
                .block()
                .stream()
                .collect(Collectors.joining());
        
        // 获取使用的源
        List<DocumentService.SearchResult> sources = ragResponse.getSearchResultsMono()
                .block();
        
        List<String> sourceTexts = sources != null 
                ? sources.stream()
                        .map(r -> r.getSegment().text())
                        .limit(3)
                        .collect(Collectors.toList())
                : Collections.emptyList();
        
        // 2. 使用 LLM-as-a-Judge 评分
        JudgeScore score = scoreAnswerWithJudge(
                testQuestion.getQuestion(),
                ragAnswer,
                testQuestion.getGroundTruth(),
                testQuestion.getSourceSegment()
        );
        
        // 3. 构建评估结果
        EvaluationResult result = new EvaluationResult();
        result.setQuestion(testQuestion.getQuestion());
        result.setRagAnswer(ragAnswer);
        result.setGroundTruth(testQuestion.getGroundTruth());
        result.setFaithfulnessScore(score.getFaithfulness());
        result.setRelevanceScore(score.getRelevance());
        result.setJudgeReasoning(score.getReasoning());
        result.setSourcesUsed(sourceTexts);
        
        return result;
    }
    
    /**
     * Feature 3: LLM-as-a-Judge Scorer
     * 使用专门的 Judge AI 对答案进行评分
     */
    private JudgeScore scoreAnswerWithJudge(
            String question,
            String ragAnswer,
            String groundTruth,
            String sourceSegment) {
        
        try {
            String judgePrompt = String.format(
                "Question: %s\n\n" +
                "RAG Answer: %s\n\n" +
                "Ground Truth: %s\n\n" +
                "Source Context: %s\n\n" +
                "Please evaluate the RAG Answer based on:\n" +
                "1. Faithfulness: Does the RAG Answer accurately reflect the Source Context?\n" +
                "2. Relevance: Does the RAG Answer address the Question?\n\n" +
                "Respond with ONLY a JSON object in this format:\n" +
                "{\"faithfulness\": 0.85, \"relevance\": 0.90, \"reasoning\": \"Brief explanation\"}",
                question,
                ragAnswer.length() > 500 ? ragAnswer.substring(0, 500) + "..." : ragAnswer,
                groundTruth,
                sourceSegment.length() > 500 ? sourceSegment.substring(0, 500) + "..." : sourceSegment
            );
            
            Prompt judgePromptObj = new Prompt(
                    Arrays.asList(
                            new SystemMessage(JUDGE_SYSTEM_PROMPT),
                            new UserMessage(judgePrompt)
                    )
            );
            
            ChatResponse response = chatModel.call(judgePromptObj);
            String responseText = response.getResult().getOutput().getText();
            
            // 解析 JSON 响应
            JudgeScore score = objectMapper.readValue(responseText, JudgeScore.class);
            
            // 确保分数在有效范围内
            score.setFaithfulness(Math.max(0.0, Math.min(1.0, score.getFaithfulness())));
            score.setRelevance(Math.max(0.0, Math.min(1.0, score.getRelevance())));
            
            log.debug("Judge 评分 - Faithfulness: {}, Relevance: {}", 
                    score.getFaithfulness(), score.getRelevance());
            
            return score;
            
        } catch (Exception e) {
            log.error("Judge 评分失败", e);
            // 返回默认分数
            JudgeScore defaultScore = new JudgeScore();
            defaultScore.setFaithfulness(0.0);
            defaultScore.setRelevance(0.0);
            defaultScore.setReasoning("评分失败: " + e.getMessage());
            return defaultScore;
        }
    }
    
    /**
     * 创建错误结果
     */
    private EvaluationResult createErrorResult(TestQuestion testQuestion, String errorMessage) {
        EvaluationResult result = new EvaluationResult();
        result.setQuestion(testQuestion.getQuestion());
        result.setRagAnswer("错误: " + errorMessage);
        result.setGroundTruth(testQuestion.getGroundTruth());
        result.setFaithfulnessScore(0.0);
        result.setRelevanceScore(0.0);
        result.setJudgeReasoning("测试失败");
        result.setSourcesUsed(Collections.emptyList());
        return result;
    }
    
    /**
     * 生成评估报告
     * 
     * @param results 评估结果列表
     * @return 报告文件路径
     */
    public String generateReport(List<EvaluationResult> results) {
        log.info("========== 生成评估报告 ==========");
        
        if (results == null || results.isEmpty()) {
            log.warn("评估结果为空，无法生成报告");
            return null;
        }
        
        // 计算平均分数
        double avgFaithfulness = results.stream()
                .mapToDouble(EvaluationResult::getFaithfulnessScore)
                .average()
                .orElse(0.0);
        
        double avgRelevance = results.stream()
                .mapToDouble(EvaluationResult::getRelevanceScore)
                .average()
                .orElse(0.0);
        
        // 计算分数分布
        long highFaithfulness = results.stream()
                .filter(r -> r.getFaithfulnessScore() >= 0.8)
                .count();
        
        long highRelevance = results.stream()
                .filter(r -> r.getRelevanceScore() >= 0.8)
                .count();
        
        // 生成 Markdown 报告
        StringBuilder report = new StringBuilder();
        report.append("# RAG 系统评估报告\n\n");
        report.append(String.format("生成时间: %s\n\n", new Date()));
        report.append(String.format("测试问题总数: %d\n\n", results.size()));
        
        report.append("## 总体评分\n\n");
        report.append(String.format("| 指标 | 平均分 | 高分率 (≥0.8) |\n"));
        report.append(String.format("|------|--------|---------------|\n"));
        report.append(String.format("| **Faithfulness** | %.3f | %d (%.1f%%) |\n", 
                avgFaithfulness, highFaithfulness, (double) highFaithfulness / results.size() * 100));
        report.append(String.format("| **Relevance** | %.3f | %d (%.1f%%) |\n\n", 
                avgRelevance, highRelevance, (double) highRelevance / results.size() * 100));
        
        report.append("## 详细结果\n\n");
        report.append("| # | 问题 | Faithfulness | Relevance |\n");
        report.append("|---|------|--------------|-----------|\n");
        
        for (int i = 0; i < results.size(); i++) {
            EvaluationResult result = results.get(i);
            String question = result.getQuestion().length() > 50 
                    ? result.getQuestion().substring(0, 50) + "..." 
                    : result.getQuestion();
            report.append(String.format("| %d | %s | %.3f | %.3f |\n",
                    i + 1,
                    question.replace("|", "\\|"),
                    result.getFaithfulnessScore(),
                    result.getRelevanceScore()));
        }
        
        report.append("\n## 结论\n\n");
        report.append(String.format(
                "本次评估共测试了 %d 个问题。\n" +
                "- **Faithfulness 平均分**: %.3f（满分 1.0）\n" +
                "- **Relevance 平均分**: %.3f（满分 1.0）\n\n",
                results.size(), avgFaithfulness, avgRelevance));
        
        if (avgFaithfulness >= 0.8 && avgRelevance >= 0.8) {
            report.append("✅ **评估结果优秀**：系统在准确性和相关性方面表现良好。\n");
        } else if (avgFaithfulness >= 0.6 && avgRelevance >= 0.6) {
            report.append("⚠️ **评估结果良好**：系统表现尚可，但仍有改进空间。\n");
        } else {
            report.append("❌ **评估结果需要改进**：系统在准确性和相关性方面需要优化。\n");
        }
        
        // 保存报告到文件
        try {
            Path reportPath = Paths.get(REPORT_FILE);
            Files.createDirectories(reportPath.getParent());
            Files.writeString(reportPath, report.toString());
            
            log.info("评估报告已保存到: {}", REPORT_FILE);
            return REPORT_FILE;
            
        } catch (IOException e) {
            log.error("保存评估报告失败", e);
            throw new RuntimeException("保存评估报告失败", e);
        }
    }
    
    /**
     * 解析 JSON 响应（简单实现）
     */
    @SuppressWarnings("unchecked")
    private Map<String, String> parseJsonResponse(String responseText) {
        try {
            // 尝试提取 JSON 对象
            String jsonText = responseText.trim();
            if (jsonText.startsWith("```json")) {
                jsonText = jsonText.substring(7);
            }
            if (jsonText.startsWith("```")) {
                jsonText = jsonText.substring(3);
            }
            if (jsonText.endsWith("```")) {
                jsonText = jsonText.substring(0, jsonText.length() - 3);
            }
            jsonText = jsonText.trim();
            
            return objectMapper.readValue(jsonText, Map.class);
        } catch (Exception e) {
            log.warn("解析 JSON 响应失败: {}", responseText, e);
            return null;
        }
    }
    
    /**
     * 完整的评估流程：生成测试集 -> 批量测试 -> 生成报告
     * 
     * @param numQuestions 要生成的问题数量（-1 表示使用所有片段）
     * @return 评估结果列表
     */
    public List<EvaluationResult> runFullEvaluation(int numQuestions) {
        log.info("========== 开始完整评估流程 ==========");
        
        // 1. 生成测试集
        List<TestQuestion> testQuestions = generateTestSet(numQuestions);
        if (testQuestions.isEmpty()) {
            log.warn("测试集生成失败或为空");
            return Collections.emptyList();
        }
        
        // 2. 批量测试
        List<EvaluationResult> results = runBatchTest(testQuestions);
        if (results.isEmpty()) {
            log.warn("批量测试未产生结果");
            return Collections.emptyList();
        }
        
        // 3. 生成报告
        generateReport(results);
        
        log.info("========== 完整评估流程完成 ==========");
        return results;
    }
}
