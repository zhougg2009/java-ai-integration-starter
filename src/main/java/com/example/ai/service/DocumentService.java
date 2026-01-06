package com.example.ai.service;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * 文档服务类 - 负责文档加载和切分
 */
@Service
public class DocumentService {

    private static final Logger log = LoggerFactory.getLogger(DocumentService.class);
    
    private static final String EXTERNAL_DOCS_DIR = "external-docs";
    
    private List<TextSegment> chunks;
    private String fileName;

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
            this.chunks = DocumentSplitters.recursive(300, 0).split(document);
            
            log.info("========== 文档切分完成 ==========");
            log.info("切片总数: {}", chunks.size());
            log.info("========== ========== ==========");
            
            return chunks;
            
        } catch (Exception e) {
            log.error("加载或切分文档时发生错误", e);
            throw new RuntimeException("文档处理失败: " + e.getMessage(), e);
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
