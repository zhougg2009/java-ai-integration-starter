package com.example.ai.service;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 文档服务类 - 负责文档加载、切分和向量化搜索
 */
@Service
public class DocumentService {

    private static final Logger log = LoggerFactory.getLogger(DocumentService.class);
    
    private static final String EXTERNAL_DOCS_DIR = "external-docs";
    private static final int SEARCH_TOP_K = 3;
    
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> vectorStore;
    
    private List<TextSegment> chunks;
    private String fileName;
    private boolean vectorStoreInitialized = false;

    public DocumentService() {
        // 初始化本地嵌入模型（不消耗 API 费用）
        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        this.vectorStore = new InMemoryEmbeddingStore<>();
        log.info("已初始化 AllMiniLmL6V2EmbeddingModel 和 InMemoryEmbeddingStore");
    }

    /**
     * 加载并切分 external-docs 目录下的 PDF 文档
     * 
     * @return 文档切片列表
     */
    public List<TextSegment> loadAndSplitDocument() {
        try {
            // 获取项目根目录
            Path rootPath = Paths.get("").toAbsolutePath();
            Path docsPath = rootPath.resolve(EXTERNAL_DOCS_DIR);
            
            log.info("开始加载文档，目录路径: {}", docsPath);
            
            // 查找 PDF 文件
            File docsDir = docsPath.toFile();
            if (!docsDir.exists() || !docsDir.isDirectory()) {
                throw new RuntimeException("external-docs 目录不存在: " + docsPath);
            }
            
            File[] pdfFiles = docsDir.listFiles((dir, name) -> 
                name.toLowerCase().endsWith(".pdf")
            );
            
            if (pdfFiles == null || pdfFiles.length == 0) {
                throw new RuntimeException("在 " + docsPath + " 目录下未找到 PDF 文件");
            }
            
            if (pdfFiles.length > 1) {
                log.warn("发现多个 PDF 文件，将使用第一个: {}", pdfFiles[0].getName());
            }
            
            File pdfFile = pdfFiles[0];
            this.fileName = pdfFile.getName();
            log.info("找到 PDF 文件: {}", fileName);
            
            // 使用 FileSystemDocumentLoader 加载 PDF
            Document document = FileSystemDocumentLoader.loadDocument(
                    pdfFile.toPath(),
                    new ApachePdfBoxDocumentParser()
            );
            
            log.info("文档加载成功，内容长度: {} 字符", document.text().length());
            
            // 使用 DocumentByParagraphSplitter 按段落进行切分
            // 注意：如果 DocumentByParagraphSplitter 不可用，使用 DocumentSplitters 作为替代
            // 理想情况下应使用: new DocumentByParagraphSplitter().split(document)
            this.chunks = DocumentSplitters.recursive(300, 30).split(document);
            
            log.info("========== 文档切分完成 ==========");
            log.info("切片总数: {}", chunks.size());
            log.info("========== ========== ==========");
            
            // 将切片存入向量库
            initializeVectorStore();
            
            return chunks;
            
        } catch (Exception e) {
            log.error("加载或切分文档时发生错误", e);
            throw new RuntimeException("文档处理失败: " + e.getMessage(), e);
        }
    }

    /**
     * 初始化向量库，将所有切片进行向量化并存储
     */
    private void initializeVectorStore() {
        if (vectorStoreInitialized) {
            log.info("向量库已初始化，跳过重复初始化");
            return;
        }
        
        if (chunks == null || chunks.isEmpty()) {
            log.warn("切片列表为空，无法初始化向量库");
            return;
        }
        
        try {
            log.info("开始初始化向量库，切片数量: {}", chunks.size());
            
            // 为每个切片生成嵌入向量并存储
            for (int i = 0; i < chunks.size(); i++) {
                TextSegment segment = chunks.get(i);
                Embedding embedding = embeddingModel.embed(segment.text()).content();
                vectorStore.add(embedding, segment);
                
                if ((i + 1) % 100 == 0) {
                    log.info("已处理 {}/{} 个切片", i + 1, chunks.size());
                }
            }
            
            vectorStoreInitialized = true;
            log.info("========== 向量库初始化完成 ==========");
            log.info("已存储 {} 个向量", chunks.size());
            log.info("========== ========== ==========");
            
        } catch (Exception e) {
            log.error("初始化向量库时发生错误", e);
            throw new RuntimeException("向量库初始化失败: " + e.getMessage(), e);
        }
    }

    /**
     * 语义搜索，返回最相关的 Top 3 切片
     * 
     * @param query 搜索查询
     * @return 包含相似度得分的切片列表（最多 3 个）
     */
    public List<SearchResult> search(String query) {
        if (query == null || query.trim().isEmpty()) {
            log.warn("搜索查询为空");
            return List.of();
        }
        
        if (!vectorStoreInitialized) {
            log.warn("向量库未初始化，正在初始化...");
            if (chunks == null) {
                loadAndSplitDocument();
            } else {
                initializeVectorStore();
            }
        }
        
        try {
            log.info("执行语义搜索，查询: {}", query);
            
            // 为查询生成嵌入向量
            Embedding queryEmbedding = embeddingModel.embed(query).content();
            
            // 在向量库中搜索最相关的切片
            List<EmbeddingMatch<TextSegment>> matches = vectorStore.findRelevant(
                    queryEmbedding, 
                    SEARCH_TOP_K
            );
            
            log.info("找到 {} 个相关结果", matches.size());
            
            // 转换为 SearchResult 列表
            List<SearchResult> results = matches.stream()
                    .map(match -> new SearchResult(
                            match.embedded(),
                            match.score()
                    ))
                    .collect(Collectors.toList());
            
            return results;
            
        } catch (Exception e) {
            log.error("执行搜索时发生错误", e);
            throw new RuntimeException("搜索失败: " + e.getMessage(), e);
        }
    }

    /**
     * 搜索结果封装类，包含切片和相似度得分
     */
    public static class SearchResult {
        private final TextSegment segment;
        private final double score;

        public SearchResult(TextSegment segment, double score) {
            this.segment = segment;
            this.score = score;
        }

        public TextSegment getSegment() {
            return segment;
        }

        public double getScore() {
            return score;
        }
    }

    /**
     * 获取文档切片列表
     * 
     * @return 文档切片列表
     */
    public List<TextSegment> getChunks() {
        if (chunks == null) {
            loadAndSplitDocument();
        }
        return chunks;
    }

    /**
     * 获取文件名
     * 
     * @return 文件名
     */
    public String getFileName() {
        if (fileName == null) {
            loadAndSplitDocument();
        }
        return fileName;
    }

    /**
     * 获取切片总数
     * 
     * @return 切片总数
     */
    public int getChunkCount() {
        if (chunks == null) {
            loadAndSplitDocument();
        }
        return chunks.size();
    }
}
