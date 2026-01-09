package com.example.ai.controller;

import com.example.ai.service.RagEvaluationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * RAG 评估控制器
 * 提供 REST API 来触发和查看评估结果
 */
@RestController
@RequestMapping("/api/evaluation")
public class RagEvaluationController {

    private static final Logger log = LoggerFactory.getLogger(RagEvaluationController.class);
    
    private final RagEvaluationService evaluationService;
    
    public RagEvaluationController(RagEvaluationService evaluationService) {
        this.evaluationService = evaluationService;
    }
    
    /**
     * 生成测试集
     * 
     * @param numQuestions 要生成的问题数量（-1 表示使用所有片段）
     * @return 生成结果
     */
    @PostMapping("/generate-test-set")
    public ResponseEntity<Map<String, Object>> generateTestSet(
            @RequestParam(defaultValue = "10") int numQuestions) {
        
        log.info("收到生成测试集请求，问题数量: {}", numQuestions);
        
        try {
            List<RagEvaluationService.TestQuestion> questions = 
                    evaluationService.generateTestSet(numQuestions);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "测试集生成成功");
            response.put("numQuestions", questions.size());
            response.put("filePath", "external-docs/test-set.json");
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            log.error("生成测试集失败", e);
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "生成测试集失败: " + e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }
    
    /**
     * 运行批量测试
     * 
     * @return 测试结果摘要
     */
    @PostMapping("/run-batch-test")
    public ResponseEntity<Map<String, Object>> runBatchTest() {
        
        log.info("收到批量测试请求");
        
        try {
            List<RagEvaluationService.EvaluationResult> results = 
                    evaluationService.runBatchTest(null);
            
            // 计算平均分数
            double avgFaithfulness = results.stream()
                    .mapToDouble(RagEvaluationService.EvaluationResult::getFaithfulnessScore)
                    .average()
                    .orElse(0.0);
            
            double avgRelevance = results.stream()
                    .mapToDouble(RagEvaluationService.EvaluationResult::getRelevanceScore)
                    .average()
                    .orElse(0.0);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "批量测试完成");
            response.put("numResults", results.size());
            response.put("avgFaithfulness", avgFaithfulness);
            response.put("avgRelevance", avgRelevance);
            response.put("reportPath", "external-docs/evaluation_report.md");
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            log.error("批量测试失败", e);
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "批量测试失败: " + e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }
    
    /**
     * 运行完整评估流程（生成测试集 -> 批量测试 -> 生成报告）
     * 
     * @param numQuestions 要生成的问题数量（-1 表示使用所有片段）
     * @return 评估结果摘要
     */
    @PostMapping("/run-full-evaluation")
    public ResponseEntity<Map<String, Object>> runFullEvaluation(
            @RequestParam(defaultValue = "10") int numQuestions) {
        
        log.info("收到完整评估请求，问题数量: {}", numQuestions);
        
        try {
            List<RagEvaluationService.EvaluationResult> results = 
                    evaluationService.runFullEvaluation(numQuestions);
            
            // 计算平均分数
            double avgFaithfulness = results.stream()
                    .mapToDouble(RagEvaluationService.EvaluationResult::getFaithfulnessScore)
                    .average()
                    .orElse(0.0);
            
            double avgRelevance = results.stream()
                    .mapToDouble(RagEvaluationService.EvaluationResult::getRelevanceScore)
                    .average()
                    .orElse(0.0);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "完整评估流程完成");
            response.put("numResults", results.size());
            response.put("avgFaithfulness", avgFaithfulness);
            response.put("avgRelevance", avgRelevance);
            response.put("testSetPath", "external-docs/test-set.json");
            response.put("reportPath", "external-docs/evaluation_report.md");
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            log.error("完整评估失败", e);
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "完整评估失败: " + e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }
    
    /**
     * 获取评估报告内容
     * 
     * @return 报告内容
     */
    @GetMapping("/report")
    public ResponseEntity<Map<String, Object>> getReport() {
        
        try {
            java.nio.file.Path reportPath = java.nio.file.Paths.get("external-docs/evaluation_report.md");
            
            if (!java.nio.file.Files.exists(reportPath)) {
                Map<String, Object> response = new HashMap<>();
                response.put("success", false);
                response.put("message", "评估报告不存在，请先运行评估");
                return ResponseEntity.status(404).body(response);
            }
            
            String reportContent = java.nio.file.Files.readString(reportPath);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("report", reportContent);
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            log.error("读取评估报告失败", e);
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "读取评估报告失败: " + e.getMessage());
            return ResponseEntity.status(500).body(response);
        }
    }
}
