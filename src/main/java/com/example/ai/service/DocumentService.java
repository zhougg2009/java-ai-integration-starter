package com.example.ai.service;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 文档服务类 - 负责文档加载、切分和向量化搜索
 */
@Service
public class DocumentService {

    private static final Logger log = LoggerFactory.getLogger(DocumentService.class);
    
    private static final String EXTERNAL_DOCS_DIR = "external-docs";
    private static final String VECTOR_STORE_FILE = "vector-store.json";
    private static final int SEARCH_TOP_K = 3; // 默认返回数量（用于向后兼容）
    
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> vectorStore;
    private final ObjectMapper objectMapper;
    
    private List<TextSegment> chunks;
    private String fileName;
    private boolean vectorStoreInitialized = false;
    private Path vectorStorePath;
    private List<Embedding> cachedEmbeddings; // 缓存 embeddings 以便保存

    public DocumentService() {
        // 初始化本地嵌入模型（不消耗 API 费用）
        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        this.vectorStore = new InMemoryEmbeddingStore<>();
        this.objectMapper = new ObjectMapper();
        
        // 设置向量库文件路径
        Path rootPath = Paths.get("").toAbsolutePath();
        this.vectorStorePath = rootPath.resolve(EXTERNAL_DOCS_DIR).resolve(VECTOR_STORE_FILE);
        
        log.info("已初始化 AllMiniLmL6V2EmbeddingModel 和 InMemoryEmbeddingStore");
        log.info("向量库持久化路径: {}", vectorStorePath);
        
        // 尝试从文件加载向量库
        initializeFromCacheOrCreate();
    }

    /**
     * 初始化向量库：优先从缓存文件加载，如果不存在则创建新的
     */
    private void initializeFromCacheOrCreate() {
        try {
            File vectorStoreFile = vectorStorePath.toFile();
            
            if (vectorStoreFile.exists() && vectorStoreFile.isFile()) {
                // 文件存在，尝试加载
                log.info("发现向量库缓存文件，尝试加载...");
                if (loadVectorStoreFromFile()) {
                    log.info("已从本地文件加载向量库");
                    vectorStoreInitialized = true;
                    return;
                } else {
                    // 加载失败，删除损坏的文件
                    log.warn("向量库文件损坏，将重新创建");
                    try {
                        Files.delete(vectorStorePath);
                    } catch (IOException e) {
                        log.warn("无法删除损坏的向量库文件: {}", e.getMessage());
                    }
                }
            }
            
            // 文件不存在或加载失败，执行完整的加载和切分流程
            log.info("向量库缓存文件不存在或已损坏，开始创建新的向量库...");
            loadAndSplitDocument();
            saveVectorStoreToFile();
            log.info("已创建新的向量库并保存到本地文件");
            
        } catch (Exception e) {
            log.error("初始化向量库时发生错误，将回退到重新处理 PDF", e);
            // 发生错误时，尝试重新处理 PDF
            try {
                loadAndSplitDocument();
                saveVectorStoreToFile();
            } catch (Exception ex) {
                log.error("重新处理 PDF 时也发生错误", ex);
                throw new RuntimeException("向量库初始化失败: " + ex.getMessage(), ex);
            }
        }
    }

    /**
     * 从文件加载向量库
     * 
     * @return 是否成功加载
     */
    private boolean loadVectorStoreFromFile() {
        try {
            File vectorStoreFile = vectorStorePath.toFile();
            if (!vectorStoreFile.exists()) {
                return false;
            }
            
            // 读取 JSON 文件
            VectorStoreData data = objectMapper.readValue(vectorStoreFile, VectorStoreData.class);
            
            // 验证数据完整性
            if (data == null || data.getChunks() == null || data.getEmbeddings() == null) {
                log.warn("向量库数据不完整");
                return false;
            }
            
            if (data.getChunks().size() != data.getEmbeddings().size()) {
                log.warn("向量库数据不一致：chunks 数量 ({}) 与 embeddings 数量 ({}) 不匹配", 
                        data.getChunks().size(), data.getEmbeddings().size());
                return false;
            }
            
            // 恢复 chunks 和 fileName
            this.chunks = data.getChunks().stream()
                    .map(chunkData -> TextSegment.from(chunkData.getText()))
                    .collect(Collectors.toList());
            this.fileName = data.getFileName();
            
            // 恢复向量库
            for (int i = 0; i < data.getChunks().size(); i++) {
                double[] embeddingArray = data.getEmbeddings().get(i);
                // 将 double[] 转换为 List<Float>（Embedding 使用 List<Float>）
                List<Float> embeddingList = new ArrayList<>();
                for (double value : embeddingArray) {
                    embeddingList.add((float) value);
                }
                Embedding embedding = Embedding.from(embeddingList);
                TextSegment segment = chunks.get(i);
                vectorStore.add(embedding, segment);
            }
            
            log.info("成功加载向量库：{} 个切片，文件名: {}", chunks.size(), fileName);
            return true;
            
        } catch (Exception e) {
            log.error("从文件加载向量库时发生错误", e);
            return false;
        }
    }

    /**
     * 保存向量库到文件
     */
    private void saveVectorStoreToFile() {
        if (chunks == null || chunks.isEmpty() || !vectorStoreInitialized) {
            log.warn("向量库未初始化或为空，跳过保存");
            return;
        }
        
        try {
            // 确保目录存在
            Files.createDirectories(vectorStorePath.getParent());
            
            // 收集所有 embeddings 和 chunks
            List<double[]> embeddings = new ArrayList<>();
            List<ChunkData> chunkDataList = new ArrayList<>();
            
            // 使用缓存的 embeddings（如果可用），否则重新计算
            List<Embedding> embeddingsToSave = cachedEmbeddings != null && cachedEmbeddings.size() == chunks.size() 
                    ? cachedEmbeddings 
                    : chunks.stream()
                            .map(segment -> embeddingModel.embed(segment.text()).content())
                            .collect(Collectors.toList());
            
            for (int i = 0; i < chunks.size(); i++) {
                Embedding embedding = embeddingsToSave.get(i);
                // 将 float 列表转换为 double 数组
                List<Float> floatList = embedding.vectorAsList();
                double[] doubleArray = new double[floatList.size()];
                for (int j = 0; j < floatList.size(); j++) {
                    doubleArray[j] = floatList.get(j).doubleValue();
                }
                embeddings.add(doubleArray);
                chunkDataList.add(new ChunkData(chunks.get(i).text()));
            }
            
            // 创建数据对象
            VectorStoreData data = new VectorStoreData(fileName, chunkDataList, embeddings);
            
            // 保存到文件
            objectMapper.writerWithDefaultPrettyPrinter()
                    .writeValue(vectorStorePath.toFile(), data);
            
            log.info("向量库已保存到文件: {}", vectorStorePath);
            
        } catch (Exception e) {
            log.error("保存向量库到文件时发生错误", e);
            // 不抛出异常，允许系统继续运行
        }
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
            
            // 将切片存入向量库（如果尚未初始化）
            if (!vectorStoreInitialized) {
                initializeVectorStore();
            }
            
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
            cachedEmbeddings = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                TextSegment segment = chunks.get(i);
                Embedding embedding = embeddingModel.embed(segment.text()).content();
                vectorStore.add(embedding, segment);
                cachedEmbeddings.add(embedding); // 缓存 embedding 以便后续保存
                
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
    /**
     * 执行语义搜索，返回最相关的切片（默认返回 3 个）
     * 
     * @param query 搜索查询
     * @return 搜索结果列表，按相似度得分降序排列
     */
    public List<SearchResult> search(String query) {
        return search(query, SEARCH_TOP_K);
    }
    
    /**
     * 执行语义搜索，返回指定数量的最相关切片
     * 
     * @param query 搜索查询
     * @param topK 返回的切片数量
     * @return 搜索结果列表，按相似度得分降序排列
     */
    public List<SearchResult> search(String query, int topK) {
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
            log.info("执行语义搜索，查询: {}，返回数量: {}", query, topK);
            
            // 为查询生成嵌入向量
            Embedding queryEmbedding = embeddingModel.embed(query).content();
            
            // 在向量库中搜索最相关的切片
            List<EmbeddingMatch<TextSegment>> matches = vectorStore.findRelevant(
                    queryEmbedding, 
                    topK
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

    /**
     * 向量库数据序列化类
     */
    private static class VectorStoreData {
        private String fileName;
        private List<ChunkData> chunks;
        private List<double[]> embeddings;

        @JsonCreator
        public VectorStoreData(
                @JsonProperty("fileName") String fileName,
                @JsonProperty("chunks") List<ChunkData> chunks,
                @JsonProperty("embeddings") List<double[]> embeddings) {
            this.fileName = fileName;
            this.chunks = chunks;
            this.embeddings = embeddings;
        }

        public String getFileName() {
            return fileName;
        }

        public List<ChunkData> getChunks() {
            return chunks;
        }

        public List<double[]> getEmbeddings() {
            return embeddings;
        }
    }

    /**
     * 切片数据序列化类
     */
    private static class ChunkData {
        private String text;

        @JsonCreator
        public ChunkData(@JsonProperty("text") String text) {
            this.text = text;
        }

        public String getText() {
            return text;
        }
    }
}
