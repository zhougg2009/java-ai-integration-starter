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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
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
    
    private List<TextSegment> chunks; // 保留用于向后兼容
    private List<TextSegment> childSegments; // 子片段（用于向量搜索）
    private Map<String, TextSegment> parentSegments; // 父片段映射 (parentId -> parent segment)
    private String fileName;
    private boolean vectorStoreInitialized = false;
    private Path vectorStorePath;
    private List<Embedding> cachedEmbeddings; // 缓存 embeddings 以便保存
    
    // Parent-Child 配置
    private static final int PARENT_SEGMENT_SIZE = 800; // 父片段大小（字符）
    private static final int CHILD_SEGMENT_SIZE = 150; // 子片段大小（字符）
    
    // 结构化模式检测（用于元数据增强）
    private static final Pattern ITEM_PATTERN = Pattern.compile(
        "(?i)\\b(Item\\s+\\d+[:\\.]?|条目\\s*\\d+[：。]?)", 
        Pattern.CASE_INSENSITIVE
    );
    private static final Pattern CHAPTER_PATTERN = Pattern.compile(
        "(?i)\\b(Chapter\\s+\\d+[:\\.]?|第\\s*\\d+\\s*章[：。]?)", 
        Pattern.CASE_INSENSITIVE
    );
    private static final Pattern SECTION_PATTERN = Pattern.compile(
        "(?i)\\b(Section\\s+\\d+[:\\.]?|节\\s*\\d+[：。]?)", 
        Pattern.CASE_INSENSITIVE
    );

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
            
            // 恢复子片段和父片段（注意：旧版本可能没有 parent-child 结构，需要兼容处理）
            this.childSegments = data.getChunks().stream()
                    .map(chunkData -> TextSegment.from(chunkData.getText()))
                    .collect(Collectors.toList());
            this.fileName = data.getFileName();
            
            // 尝试从子片段重建父片段（如果元数据存在）
            this.parentSegments = new HashMap<>();
            for (TextSegment child : childSegments) {
                if (child.metadata() != null && child.metadata().containsKey("parentId")) {
                    String parentId = child.metadata().get("parentId");
                    // 如果父片段还不存在，创建一个（从子片段推断）
                    if (!parentSegments.containsKey(parentId)) {
                        // 注意：从文件加载时，我们无法完全恢复父片段
                        // 这里使用子片段作为占位符，实际使用时会通过其他方式获取
                        parentSegments.put(parentId, child);
                    }
                }
            }
            
            // 如果没有父片段，说明是旧版本数据，需要重新处理
            if (parentSegments.isEmpty()) {
                log.warn("检测到旧版本向量库数据（无 Parent-Child 结构），建议重新处理 PDF");
                // 为了兼容，将子片段也作为父片段
                for (int i = 0; i < childSegments.size(); i++) {
                    parentSegments.put("parent_" + i, childSegments.get(i));
                }
            }
            
            // 恢复向量库（只存储子片段）
            for (int i = 0; i < childSegments.size(); i++) {
                double[] embeddingArray = data.getEmbeddings().get(i);
                // 将 double[] 转换为 List<Float>（Embedding 使用 List<Float>）
                List<Float> embeddingList = new ArrayList<>();
                for (double value : embeddingArray) {
                    embeddingList.add((float) value);
                }
                Embedding embedding = Embedding.from(embeddingList);
                TextSegment segment = childSegments.get(i);
                vectorStore.add(embedding, segment);
            }
            
            // 为了向后兼容，保留 chunks
            this.chunks = new ArrayList<>(childSegments);
            
            log.info("成功加载向量库：{} 个子片段，{} 个父片段，文件名: {}", 
                    childSegments.size(), parentSegments.size(), fileName);
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
        if (childSegments == null || childSegments.isEmpty() || !vectorStoreInitialized) {
            log.warn("向量库未初始化或为空，跳过保存");
            return;
        }
        
        try {
            // 确保目录存在
            Files.createDirectories(vectorStorePath.getParent());
            
            // 收集所有 embeddings 和 child segments（只保存子片段用于向量搜索）
            List<double[]> embeddings = new ArrayList<>();
            List<ChunkData> chunkDataList = new ArrayList<>();
            
            // 使用缓存的 embeddings（如果可用），否则重新计算
            List<Embedding> embeddingsToSave = cachedEmbeddings != null && cachedEmbeddings.size() == childSegments.size() 
                    ? cachedEmbeddings 
                    : childSegments.stream()
                            .map(segment -> embeddingModel.embed(segment.text()).content())
                            .collect(Collectors.toList());
            
            for (int i = 0; i < childSegments.size(); i++) {
                Embedding embedding = embeddingsToSave.get(i);
                // 将 float 列表转换为 double 数组
                List<Float> floatList = embedding.vectorAsList();
                double[] doubleArray = new double[floatList.size()];
                for (int j = 0; j < floatList.size(); j++) {
                    doubleArray[j] = floatList.get(j).doubleValue();
                }
                embeddings.add(doubleArray);
                
                // 保存子片段文本和元数据（包含 parentId）
                TextSegment childSegment = childSegments.get(i);
                String childText = childSegment.text();
                // 如果元数据中有 parentId，将其附加到文本中（用于恢复）
                if (childSegment.metadata() != null && childSegment.metadata().containsKey("parentId")) {
                    String parentId = childSegment.metadata().get("parentId");
                    // 使用特殊标记保存元数据信息
                    childText = "<!--PARENT_ID:" + parentId + "--> " + childText;
                }
                chunkDataList.add(new ChunkData(childText));
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
            
            // 使用 Parent-Child 策略进行切分
            log.info("开始使用 Parent-Child 策略切分文档...");
            createParentChildSegments(document);
            
            log.info("========== 文档切分完成 ==========");
            log.info("父片段总数: {}", parentSegments.size());
            log.info("子片段总数: {}", childSegments.size());
            log.info("========== ========== ==========");
            
            // 为了向后兼容，保留 chunks（使用 child segments）
            this.chunks = new ArrayList<>(childSegments);
            
            // 将子片段存入向量库（如果尚未初始化）
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
     * 创建 Parent-Child 片段结构
     * - Parent segments: 大片段（800 字符）包含完整上下文
     * - Child segments: 小片段（150 字符）用于高精度向量搜索
     * 
     * @param document 原始文档
     */
    private void createParentChildSegments(Document document) {
        this.childSegments = new ArrayList<>();
        this.parentSegments = new HashMap<>();
        
        log.info("开始创建 Parent-Child 片段结构...");
        log.info("父片段大小: {} 字符，子片段大小: {} 字符", PARENT_SEGMENT_SIZE, CHILD_SEGMENT_SIZE);
        
        // 首先创建父片段（大片段）
        List<TextSegment> tempParentSegments = DocumentSplitters.recursive(PARENT_SEGMENT_SIZE, 50).split(document);
        
        int parentIndex = 0;
        for (TextSegment parentSegment : tempParentSegments) {
            String parentId = "parent_" + parentIndex;
            String parentText = parentSegment.text();
            
            // 检测并提取结构化元数据（Item, Chapter, Section）
            Map<String, String> metadata = extractStructuralMetadata(parentText);
            
            // 创建带元数据的父片段
            TextSegment enrichedParentSegment = TextSegment.from(parentText);
            enrichedParentSegment.metadata().put("parentId", parentId);
            enrichedParentSegment.metadata().put("parentIndex", String.valueOf(parentIndex));
            // 添加结构化元数据
            metadata.forEach((key, value) -> enrichedParentSegment.metadata().put(key, value));
            
            // 存储父片段
            parentSegments.put(parentId, enrichedParentSegment);
            
            // 从父片段中创建子片段（小片段，用于向量搜索）
            int parentLength = parentText.length();
            int childIndex = 0;
            
            // 使用滑动窗口创建子片段（有重叠以确保覆盖）
            int overlap = 30; // 子片段之间的重叠
            int start = 0;
            
            while (start < parentLength) {
                int end = Math.min(start + CHILD_SEGMENT_SIZE, parentLength);
                String childText = parentText.substring(start, end);
                
                // 创建子片段，并在元数据中存储父片段 ID 和结构化元数据
                TextSegment childSegment = TextSegment.from(childText);
                childSegment.metadata().put("parentId", parentId);
                childSegment.metadata().put("parentIndex", String.valueOf(parentIndex));
                childSegment.metadata().put("childIndex", String.valueOf(childIndex));
                // 子片段继承父片段的结构化元数据
                metadata.forEach((key, value) -> childSegment.metadata().put(key, value));
                
                childSegments.add(childSegment);
                
                // 移动到下一个子片段（考虑重叠）
                start += (CHILD_SEGMENT_SIZE - overlap);
                childIndex++;
            }
            
            parentIndex++;
            
            if (parentIndex % 50 == 0) {
                log.info("已处理 {}/{} 个父片段", parentIndex, tempParentSegments.size());
            }
        }
        
        log.info("Parent-Child 片段创建完成：{} 个父片段，{} 个子片段", 
                parentSegments.size(), childSegments.size());
    }
    
    /**
     * 提取结构化元数据（Item, Chapter, Section 等）
     * 
     * @param text 文本内容
     * @return 元数据映射
     */
    private Map<String, String> extractStructuralMetadata(String text) {
        Map<String, String> metadata = new HashMap<>();
        
        // 检测 Item（如 "Item 1:", "Item 2:" 等）
        Matcher itemMatcher = ITEM_PATTERN.matcher(text);
        if (itemMatcher.find()) {
            String itemMatch = itemMatcher.group(1);
            // 提取数字
            Pattern numberPattern = Pattern.compile("\\d+");
            Matcher numberMatcher = numberPattern.matcher(itemMatch);
            if (numberMatcher.find()) {
                metadata.put("item_id", numberMatcher.group());
                metadata.put("item_label", "Item " + numberMatcher.group());
            }
        }
        
        // 检测 Chapter（如 "Chapter 1:", "Chapter 2:" 等）
        Matcher chapterMatcher = CHAPTER_PATTERN.matcher(text);
        if (chapterMatcher.find()) {
            String chapterMatch = chapterMatcher.group(1);
            Pattern numberPattern = Pattern.compile("\\d+");
            Matcher numberMatcher = numberPattern.matcher(chapterMatch);
            if (numberMatcher.find()) {
                metadata.put("chapter_id", numberMatcher.group());
                metadata.put("chapter_label", "Chapter " + numberMatcher.group());
            }
        }
        
        // 检测 Section（如 "Section 1:", "Section 2:" 等）
        Matcher sectionMatcher = SECTION_PATTERN.matcher(text);
        if (sectionMatcher.find()) {
            String sectionMatch = sectionMatcher.group(1);
            Pattern numberPattern = Pattern.compile("\\d+");
            Matcher numberMatcher = numberPattern.matcher(sectionMatch);
            if (numberMatcher.find()) {
                metadata.put("section_id", numberMatcher.group());
                metadata.put("section_label", "Section " + numberMatcher.group());
            }
        }
        
        return metadata;
    }
    
    /**
     * 根据子片段获取对应的父片段
     * 
     * @param childSegment 子片段
     * @return 父片段，如果找不到则返回 null
     */
    public TextSegment getParentSegment(TextSegment childSegment) {
        if (childSegment == null || childSegment.metadata() == null) {
            return null;
        }
        
        String parentId = childSegment.metadata().get("parentId");
        if (parentId == null) {
            return null;
        }
        
        return parentSegments.get(parentId);
    }
    
    /**
     * 初始化向量库，将所有子片段进行向量化并存储
     */
    private void initializeVectorStore() {
        if (vectorStoreInitialized) {
            log.info("向量库已初始化，跳过重复初始化");
            return;
        }
        
        if (childSegments == null || childSegments.isEmpty()) {
            log.warn("子片段列表为空，无法初始化向量库");
            return;
        }
        
        try {
            log.info("开始初始化向量库，子片段数量: {}", childSegments.size());
            
            // 只为子片段生成嵌入向量并存储（用于向量搜索）
            cachedEmbeddings = new ArrayList<>();
            for (int i = 0; i < childSegments.size(); i++) {
                TextSegment segment = childSegments.get(i);
                Embedding embedding = embeddingModel.embed(segment.text()).content();
                vectorStore.add(embedding, segment);
                cachedEmbeddings.add(embedding); // 缓存 embedding 以便后续保存
                
                if ((i + 1) % 100 == 0) {
                    log.info("已处理 {}/{} 个子片段", i + 1, childSegments.size());
                }
            }
            
            vectorStoreInitialized = true;
            log.info("========== 向量库初始化完成 ==========");
            log.info("已存储 {} 个子片段向量（用于搜索）", childSegments.size());
            log.info("已存储 {} 个父片段（用于上下文）", parentSegments.size());
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
     * 执行关键词搜索（基于文本匹配）
     * 用于混合搜索中的关键词搜索部分
     * 
     * @param query 搜索查询
     * @param topK 返回的切片数量
     * @return 搜索结果列表，按匹配度降序排列
     */
    public List<SearchResult> keywordSearch(String query, int topK) {
        if (query == null || query.trim().isEmpty()) {
            log.warn("关键词搜索查询为空");
            return List.of();
        }
        
        if (childSegments == null || childSegments.isEmpty()) {
            log.warn("子片段列表为空，无法执行关键词搜索");
            return List.of();
        }
        
        try {
            log.info("执行关键词搜索，查询: {}，返回数量: {}", query, topK);
            
            // 将查询转换为小写并分割为关键词
            String[] queryTerms = query.toLowerCase().split("\\s+");
            List<String> keywords = new ArrayList<>();
            for (String term : queryTerms) {
                // 移除标点符号
                term = term.replaceAll("[^a-zA-Z0-9]", "");
                if (!term.isEmpty() && term.length() > 2) { // 忽略太短的词
                    keywords.add(term);
                }
            }
            
            if (keywords.isEmpty()) {
                log.warn("未找到有效的关键词");
                return List.of();
            }
            
            log.debug("提取的关键词: {}", keywords);
            
            // 对每个子片段计算关键词匹配得分
            List<SearchResult> results = new ArrayList<>();
            for (TextSegment segment : childSegments) {
                String text = segment.text().toLowerCase();
                double score = calculateKeywordScore(text, keywords);
                
                if (score > 0) {
                    results.add(new SearchResult(segment, score));
                }
            }
            
            // 按得分降序排序
            results.sort((a, b) -> Double.compare(b.getScore(), a.getScore()));
            
            // 返回 topK 个结果
            List<SearchResult> topResults = results.stream()
                    .limit(topK)
                    .collect(Collectors.toList());
            
            log.info("关键词搜索完成，找到 {} 个匹配结果", topResults.size());
            return topResults;
            
        } catch (Exception e) {
            log.error("执行关键词搜索时发生错误", e);
            throw new RuntimeException("关键词搜索失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 计算关键词匹配得分
     * 使用 TF-IDF 简化版本：词频 + 位置权重
     * 
     * @param text 文本内容
     * @param keywords 关键词列表
     * @return 匹配得分
     */
    private double calculateKeywordScore(String text, List<String> keywords) {
        if (text == null || text.isEmpty() || keywords == null || keywords.isEmpty()) {
            return 0.0;
        }
        
        double totalScore = 0.0;
        int totalMatches = 0;
        
        for (String keyword : keywords) {
            // 计算关键词在文本中的出现次数
            int count = 0;
            int index = text.indexOf(keyword);
            int firstOccurrence = index;
            
            while (index >= 0) {
                count++;
                index = text.indexOf(keyword, index + keyword.length());
            }
            
            if (count > 0) {
                // 基础得分：词频（对数缩放）
                double frequencyScore = Math.log1p(count);
                
                // 位置权重：出现在文本前面的关键词得分更高
                double positionWeight = 1.0;
                if (firstOccurrence < text.length() / 4) {
                    positionWeight = 1.5; // 前25%的位置权重更高
                } else if (firstOccurrence < text.length() / 2) {
                    positionWeight = 1.2; // 前50%的位置权重稍高
                }
                
                // 精确匹配权重：如果关键词是完整单词（不是子串），得分更高
                double exactMatchWeight = 1.0;
                if (firstOccurrence >= 0) {
                    char before = firstOccurrence > 0 ? text.charAt(firstOccurrence - 1) : ' ';
                    char after = firstOccurrence + keyword.length() < text.length() 
                            ? text.charAt(firstOccurrence + keyword.length()) : ' ';
                    if (!Character.isLetterOrDigit(before) && !Character.isLetterOrDigit(after)) {
                        exactMatchWeight = 1.3; // 完整单词匹配
                    }
                }
                
                totalScore += frequencyScore * positionWeight * exactMatchWeight;
                totalMatches++;
            }
        }
        
        // 归一化得分：考虑匹配的关键词数量和文本长度
        if (totalMatches == 0) {
            return 0.0;
        }
        
        // 归一化到 0-1 范围
        double normalizedScore = totalScore / (keywords.size() * 2.0);
        return Math.min(1.0, normalizedScore);
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
