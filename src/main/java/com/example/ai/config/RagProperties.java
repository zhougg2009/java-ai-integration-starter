package com.example.ai.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * RAG 功能开关配置属性
 * 用于消融研究（Ablation Studies），允许独立开关各个优化功能
 */
@Component
@ConfigurationProperties(prefix = "rag.features")
public class RagProperties {

    /**
     * HyDE (Hypothetical Document Embeddings) 功能开关
     * 默认: true
     */
    private boolean hydeEnabled = true;

    /**
     * Step-back Prompting 功能开关
     * 默认: true
     */
    private boolean stepbackEnabled = true;

    /**
     * Reranking 功能开关
     * 默认: true
     */
    private boolean rerankEnabled = true;

    /**
     * Hybrid Search (向量搜索 + 关键词搜索) 功能开关
     * 默认: true
     * 如果关闭，将只使用向量搜索
     */
    private boolean hybridSearchEnabled = true;

    // Getters and Setters
    public boolean isHydeEnabled() {
        return hydeEnabled;
    }

    public void setHydeEnabled(boolean hydeEnabled) {
        this.hydeEnabled = hydeEnabled;
    }

    public boolean isStepbackEnabled() {
        return stepbackEnabled;
    }

    public void setStepbackEnabled(boolean stepbackEnabled) {
        this.stepbackEnabled = stepbackEnabled;
    }

    public boolean isRerankEnabled() {
        return rerankEnabled;
    }

    public void setRerankEnabled(boolean rerankEnabled) {
        this.rerankEnabled = rerankEnabled;
    }

    public boolean isHybridSearchEnabled() {
        return hybridSearchEnabled;
    }

    public void setHybridSearchEnabled(boolean hybridSearchEnabled) {
        this.hybridSearchEnabled = hybridSearchEnabled;
    }

    @Override
    public String toString() {
        return "RagProperties{" +
                "hydeEnabled=" + hydeEnabled +
                ", stepbackEnabled=" + stepbackEnabled +
                ", rerankEnabled=" + rerankEnabled +
                ", hybridSearchEnabled=" + hybridSearchEnabled +
                '}';
    }
}
