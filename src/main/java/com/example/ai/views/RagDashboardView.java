package com.example.ai.views;

import com.example.ai.service.DocumentService;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.theme.lumo.LumoUtility;
import dev.langchain4j.data.segment.TextSegment;

import java.util.List;
import java.util.stream.Collectors;

/**
 * RAG 信息仪表板视图
 * 展示文档加载和切分信息
 */
@Route("rag-info")
@PageTitle("RAG Dashboard")
public class RagDashboardView extends VerticalLayout {

    private final DocumentService documentService;

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
        headerInfo.addClassName(LumoUtility.Margin.Bottom.LARGE);
        
        // 创建 Grid 展示前 10 个切片
        Grid<TextSegment> chunkGrid = new Grid<>(TextSegment.class, false);
        chunkGrid.setWidthFull();
        chunkGrid.setMaxWidth("1200px");
        
        // 配置列
        chunkGrid.addColumn(chunk -> {
            int index = chunks.indexOf(chunk) + 1;
            return "切片 #" + index;
        }).setHeader("序号").setWidth("100px").setFlexGrow(0);
        
        chunkGrid.addColumn(chunk -> {
            String text = chunk.text();
            // 限制预览长度，超过 200 字符显示省略号
            if (text.length() > 200) {
                return text.substring(0, 200) + "...";
            }
            return text;
        }).setHeader("内容预览").setAutoWidth(true);
        
        chunkGrid.addColumn(chunk -> chunk.text().length() + " 字符")
                .setHeader("长度")
                .setWidth("100px")
                .setFlexGrow(0);
        
        // 设置数据源（前 10 个切片）
        List<TextSegment> previewChunks = chunks.stream()
                .limit(10)
                .collect(Collectors.toList());
        chunkGrid.setItems(previewChunks);
        
        // 添加标题说明
        Span gridTitle = new Span("前 10 个切片预览");
        gridTitle.addClassName(LumoUtility.FontSize.MEDIUM);
        gridTitle.addClassName(LumoUtility.FontWeight.SEMIBOLD);
        gridTitle.addClassName(LumoUtility.Margin.Bottom.SMALL);
        
        // 布局组件
        VerticalLayout contentLayout = new VerticalLayout();
        contentLayout.setWidthFull();
        contentLayout.setMaxWidth("1200px");
        contentLayout.setAlignItems(Alignment.STRETCH);
        contentLayout.setPadding(false);
        contentLayout.setSpacing(true);
        
        contentLayout.add(gridTitle, chunkGrid);
        
        add(headerInfo, contentLayout);
        setFlexGrow(1, contentLayout);
    }
}
