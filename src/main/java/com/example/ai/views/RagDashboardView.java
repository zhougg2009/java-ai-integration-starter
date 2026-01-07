package com.example.ai.views;

import com.example.ai.service.DocumentService;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.Icon;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.data.renderer.ComponentRenderer;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.theme.lumo.LumoUtility;
import dev.langchain4j.data.segment.TextSegment;

import java.util.List;
import java.util.stream.Collectors;

/**
 * RAG 信息仪表板视图
 * 展示文档加载、切分信息和语义搜索功能
 */
@Route("rag-info")
@PageTitle("RAG Dashboard")
public class RagDashboardView extends VerticalLayout {

    private final DocumentService documentService;
    private final TextField searchField;
    private final Button searchButton;
    private final Span scoreInfo;
    private final Grid<TextSegment> chunkGrid;
    private final Span gridTitle;

    public RagDashboardView(DocumentService documentService) {
        this.documentService = documentService;
        
        setSizeFull();
        setPadding(true);
        setSpacing(true);
        setAlignItems(Alignment.CENTER);
        addClassName("rag-dashboard-view");
        
        // 加载文档数据
        String fileName = documentService.getFileName();
        int chunkCount = documentService.getChunkCount();
        List<TextSegment> chunks = documentService.getChunks();
        
        // 创建顶部信息展示
        Span headerInfo = new Span();
        headerInfo.setText(String.format("文件名: %s | 切片总数: %d", fileName, chunkCount));
        headerInfo.addClassName(LumoUtility.FontSize.LARGE);
        headerInfo.addClassName(LumoUtility.FontWeight.BOLD);
        headerInfo.addClassName(LumoUtility.Margin.Bottom.MEDIUM);
        
        // 创建搜索框
        searchField = new TextField();
        searchField.setPlaceholder("输入搜索关键词...");
        searchField.setWidthFull();
        searchField.setClearButtonVisible(true);
        
        // 创建搜索按钮
        searchButton = new Button("搜索");
        searchButton.addClassName(LumoUtility.Margin.Left.SMALL);
        searchButton.addClickListener(e -> performSearch());
        
        // 支持回车键搜索
        searchField.addKeyPressListener(e -> {
            if (e.getKey().equals("Enter")) {
                performSearch();
            }
        });
        
        // 也支持 KeyDownEvent 确保回车键触发搜索
        searchField.addKeyDownListener(e -> {
            if (e.getKey().equals("Enter")) {
                performSearch();
            }
        });
        
        // 创建搜索区域布局
        HorizontalLayout searchLayout = new HorizontalLayout();
        searchLayout.setWidthFull();
        searchLayout.setMaxWidth("1200px");
        searchLayout.setAlignItems(Alignment.CENTER);
        searchLayout.setSpacing(true);
        searchLayout.add(searchField, searchButton);
        searchLayout.setFlexGrow(1, searchField);
        
        // 创建相似度得分显示
        scoreInfo = new Span();
        scoreInfo.addClassName(LumoUtility.FontSize.SMALL);
        scoreInfo.addClassName(LumoUtility.TextColor.SECONDARY);
        scoreInfo.addClassName(LumoUtility.Margin.Bottom.SMALL);
        scoreInfo.setVisible(false);
        
        // 创建 Grid 展示切片
        chunkGrid = new Grid<>(TextSegment.class, false);
        chunkGrid.setWidthFull();
        chunkGrid.setMaxWidth("1200px");
        
        // 启用选择模式，支持 Ctrl+C 复制
        chunkGrid.setSelectionMode(Grid.SelectionMode.SINGLE);
        
        // 配置列 - 序号列
        chunkGrid.addColumn(chunk -> {
            // 获取当前数据视图中的所有项目来计算索引
            var dataView = chunkGrid.getListDataView();
            if (dataView != null) {
                var items = dataView.getItems().toList();
                int index = items.indexOf(chunk);
                return index >= 0 ? "结果 #" + (index + 1) : "结果 #1";
            }
            return "结果 #1";
        }).setHeader("序号").setWidth("100px").setFlexGrow(0);
        
        // 配置列 - 内容预览列（使用 ComponentRenderer 支持文本换行）
        chunkGrid.addColumn(new ComponentRenderer<>(chunk -> {
            String text = chunk.text();
            // 限制预览长度，超过 150 字符显示省略号
            String displayText = text.length() > 150 ? text.substring(0, 150) + "..." : text;
            
            Span span = new Span(displayText);
            span.getStyle().set("white-space", "normal");
            span.getStyle().set("word-break", "break-all");
            span.getStyle().set("display", "block");
            return span;
        })).setHeader("内容预览").setFlexGrow(1);
        
        // 配置列 - 长度列
        chunkGrid.addColumn(chunk -> chunk.text().length() + " 字符")
                .setHeader("长度")
                .setWidth("100px")
                .setFlexGrow(0);
        
        // 配置列 - Action 列（复制按钮）
        chunkGrid.addColumn(new ComponentRenderer<>(chunk -> {
            Icon copyIcon = VaadinIcon.COPY.create();
            copyIcon.setSize("16px");
            copyIcon.getStyle().set("cursor", "pointer");
            copyIcon.getStyle().set("color", "var(--lumo-primary-color)");
            copyIcon.getStyle().set("transition", "color 0.2s");
            
            // 添加点击事件处理
            copyIcon.addClickListener(e -> {
                // 添加视觉反馈
                copyIcon.getStyle().set("color", "var(--lumo-success-color)");
                // 执行复制
                copyToClipboard(chunk.text());
                // 恢复颜色
                getUI().ifPresent(ui -> {
                    ui.access(() -> {
                        ui.getPage().executeJs(
                            "setTimeout(function() { $0.style.setProperty('color', 'var(--lumo-primary-color)'); }, 300);",
                            copyIcon.getElement()
                        );
                    });
                });
            });
            
            copyIcon.getElement().setAttribute("title", "点击复制内容");
            
            // 添加悬停效果（通过 JavaScript）
            copyIcon.getElement().executeJs(
                "this.addEventListener('mouseenter', function() { this.style.setProperty('color', 'var(--lumo-success-color)'); });" +
                "this.addEventListener('mouseleave', function() { this.style.setProperty('color', 'var(--lumo-primary-color)'); });"
            );
            
            return copyIcon;
        })).setHeader("Action").setWidth("80px").setFlexGrow(0);
        
        // 添加 Ctrl+C 快捷键支持
        // 在 Java 端监听选择变化，将选中项的文本存储到 Grid 元素属性中
        chunkGrid.addSelectionListener(e -> {
            e.getFirstSelectedItem().ifPresent(item -> {
                // 将选中的文本存储到 Grid 元素的属性中，供 JavaScript 使用
                chunkGrid.getElement().setProperty("selectedText", item.text());
            });
        });
        
        // 通过 JavaScript 监听键盘事件，复制选中行的文本
        chunkGrid.getElement().executeJs(
            "const grid = this;" +
            "document.addEventListener('keydown', function(e) {" +
            "  // 检查事件是否发生在 Grid 或其子元素上" +
            "  if (grid.contains(e.target) || grid === e.target) {" +
            "    if ((e.ctrlKey || e.metaKey) && e.key === 'c') {" +
            "      const selectedText = grid.getProperty('selectedText');" +
            "      if (selectedText) {" +
            "        if (navigator.clipboard && navigator.clipboard.writeText) {" +
            "          navigator.clipboard.writeText(selectedText).catch(function(err) {" +
            "            console.error('复制失败:', err);" +
            "          });" +
            "        } else {" +
            "          const textarea = document.createElement('textarea');" +
            "          textarea.value = selectedText;" +
            "          textarea.style.position = 'fixed';" +
            "          textarea.style.opacity = '0';" +
            "          document.body.appendChild(textarea);" +
            "          textarea.select();" +
            "          document.execCommand('copy');" +
            "          document.body.removeChild(textarea);" +
            "        }" +
            "        e.preventDefault();" +
            "        e.stopPropagation();" +
            "      }" +
            "    }" +
            "  }" +
            "});"
        );
        
        // 初始化显示前 10 个切片
        gridTitle = new Span("前 10 个切片预览");
        gridTitle.addClassName(LumoUtility.FontSize.MEDIUM);
        gridTitle.addClassName(LumoUtility.FontWeight.SEMIBOLD);
        gridTitle.addClassName(LumoUtility.Margin.Bottom.SMALL);
        
        List<TextSegment> previewChunks = chunks.stream()
                .limit(10)
                .collect(Collectors.toList());
        chunkGrid.setItems(previewChunks);
        
        // 布局组件
        VerticalLayout contentLayout = new VerticalLayout();
        contentLayout.setWidthFull();
        contentLayout.setMaxWidth("1200px");
        contentLayout.setAlignItems(Alignment.STRETCH);
        contentLayout.setPadding(false);
        contentLayout.setSpacing(true);
        
        contentLayout.add(gridTitle, chunkGrid);
        
        add(headerInfo, searchLayout, scoreInfo, contentLayout);
        setFlexGrow(1, contentLayout);
    }

    /**
     * 执行搜索操作
     */
    private void performSearch() {
        String query = searchField.getValue();
        
        if (query == null || query.trim().isEmpty()) {
            // 如果搜索框为空，恢复显示前 10 个切片
            List<TextSegment> chunks = documentService.getChunks();
            List<TextSegment> previewChunks = chunks.stream()
                    .limit(10)
                    .collect(Collectors.toList());
            chunkGrid.setItems(previewChunks);
            gridTitle.setText("前 10 个切片预览");
            scoreInfo.setVisible(false);
            return;
        }
        
        try {
            // 执行语义搜索
            List<DocumentService.SearchResult> results = documentService.search(query);
            
            if (results.isEmpty()) {
                chunkGrid.setItems(List.of());
                gridTitle.setText("未找到相关结果");
                scoreInfo.setVisible(false);
                return;
            }
            
            // 更新 Grid 显示搜索结果
            List<TextSegment> resultSegments = results.stream()
                    .map(DocumentService.SearchResult::getSegment)
                    .collect(Collectors.toList());
            chunkGrid.setItems(resultSegments);
            
            // 更新标题
            gridTitle.setText(String.format("搜索结果 (Top %d)", results.size()));
            
            // 显示相似度得分
            if (!results.isEmpty()) {
                StringBuilder scoreText = new StringBuilder("相似度得分: ");
                for (int i = 0; i < results.size(); i++) {
                    DocumentService.SearchResult result = results.get(i);
                    scoreText.append(String.format("结果 #%d: %.4f", i + 1, result.getScore()));
                    if (i < results.size() - 1) {
                        scoreText.append(" | ");
                    }
                }
                scoreInfo.setText(scoreText.toString());
                scoreInfo.setVisible(true);
            } else {
                scoreInfo.setVisible(false);
            }
            
        } catch (Exception e) {
            // 显示错误信息
            chunkGrid.setItems(List.of());
            gridTitle.setText("搜索失败: " + e.getMessage());
            scoreInfo.setVisible(false);
        }
    }

    /**
     * 复制文本到剪贴板
     * 
     * @param text 要复制的文本
     */
    private void copyToClipboard(String text) {
        if (text == null || text.isEmpty()) {
            return;
        }
        
        // 确保在 UI 线程中执行
        getUI().ifPresent(ui -> {
            ui.access(() -> {
                // 使用更简单、更可靠的 JavaScript 代码
                String jsCode = 
                    "var text = " + escapeForJs(text) + ";" +
                    "var fallbackCopy = function() {" +
                    "  var textarea = document.createElement('textarea');" +
                    "  textarea.value = text;" +
                    "  textarea.style.position = 'fixed';" +
                    "  textarea.style.left = '-9999px';" +
                    "  textarea.style.top = '0';" +
                    "  document.body.appendChild(textarea);" +
                    "  textarea.focus();" +
                    "  textarea.select();" +
                    "  try {" +
                    "    var successful = document.execCommand('copy');" +
                    "    document.body.removeChild(textarea);" +
                    "    if (successful) {" +
                    "      showNotification();" +
                    "      return true;" +
                    "    }" +
                    "    return false;" +
                    "  } catch (err) {" +
                    "    document.body.removeChild(textarea);" +
                    "    console.error('复制失败:', err);" +
                    "    return false;" +
                    "  }" +
                    "};" +
                    "var showNotification = function() {" +
                    "  var notification = document.createElement('div');" +
                    "  notification.textContent = '已复制到剪贴板';" +
                    "  notification.style.cssText = 'position:fixed;top:20px;right:20px;background:#4CAF50;color:white;padding:12px 24px;border-radius:4px;z-index:10000;box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:system-ui;';" +
                    "  document.body.appendChild(notification);" +
                    "  setTimeout(function() { if (notification.parentNode) notification.parentNode.removeChild(notification); }, 2000);" +
                    "};" +
                    "if (navigator.clipboard && navigator.clipboard.writeText) {" +
                    "  navigator.clipboard.writeText(text).then(function() {" +
                    "    showNotification();" +
                    "  }).catch(function(err) {" +
                    "    console.error('Clipboard API 失败，尝试备用方法:', err);" +
                    "    fallbackCopy();" +
                    "  });" +
                    "} else {" +
                    "  fallbackCopy();" +
                    "}";
                
                ui.getPage().executeJs(jsCode);
            });
        });
    }
    
    /**
     * 将 Java 字符串转义为 JavaScript 字符串字面量
     * 
     * @param text 要转义的文本
     * @return 转义后的 JavaScript 字符串字面量
     */
    private String escapeForJs(String text) {
        if (text == null) {
            return "null";
        }
        // 使用 JSON 格式转义，这是最安全的方式
        StringBuilder sb = new StringBuilder("\"");
        for (char c : text.toCharArray()) {
            switch (c) {
                case '"':
                    sb.append("\\\"");
                    break;
                case '\\':
                    sb.append("\\\\");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                case '\b':
                    sb.append("\\b");
                    break;
                case '\f':
                    sb.append("\\f");
                    break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                    break;
            }
        }
        sb.append("\"");
        return sb.toString();
    }
}
