package com.example.ai.views;

import com.example.ai.service.ChatService;
import com.example.ai.service.DocumentService;
import com.vaadin.flow.component.Component;
import com.vaadin.flow.component.details.Details;
import com.vaadin.flow.component.details.DetailsVariant;
import com.vaadin.flow.component.html.*;
import com.vaadin.flow.component.messages.MessageInput;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.progressbar.ProgressBar;
import com.vaadin.flow.component.tabs.Tab;
import com.vaadin.flow.component.tabs.Tabs;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.theme.lumo.LumoUtility;
import dev.langchain4j.data.segment.TextSegment;
import jakarta.annotation.PreDestroy;
import org.commonmark.node.Node;
import org.commonmark.parser.Parser;
import org.commonmark.renderer.html.HtmlRenderer;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Vaadin 聊天视图
 * 该视图仅负责 UI 展示和用户交互，所有 AI 逻辑通过 ChatService 处理
 */
@Route("")
@PageTitle("Java AI Integration Lab")
public class ChatView extends VerticalLayout {

    private final ChatService chatService;
    private final MessageInput messageInput;
    private final VerticalLayout chatContainer;
    private final Div chatScroller;
    private final AtomicReference<String> currentAiResponse;
    private final AtomicReference<Component> currentAiMessageComponent;
    private final AtomicReference<Boolean> pendingUpdate;
    private final ScheduledExecutorService updateScheduler;
    private ProgressBar loadingIndicator;
    
    // Markdown 处理器
    private final Parser markdownParser;
    private final HtmlRenderer htmlRenderer;
    
    // 模式切换
    private Tabs modeTabs;
    private Tab generalChatTab;
    private Tab bookAssistantTab;
    private boolean isBookAssistantMode = false;
    
    // RAG 模式下的检索结果存储
    private final AtomicReference<List<DocumentService.SearchResult>> currentSearchResults;

    public ChatView(ChatService chatService) {
        this.chatService = chatService;
        this.currentAiResponse = new AtomicReference<>("");
        this.currentAiMessageComponent = new AtomicReference<>();
        this.pendingUpdate = new AtomicReference<>(false);
        this.updateScheduler = Executors.newSingleThreadScheduledExecutor();
        this.currentSearchResults = new AtomicReference<>(List.of());
        
        // 初始化 Markdown 处理器
        this.markdownParser = Parser.builder().build();
        this.htmlRenderer = HtmlRenderer.builder().build();

        setSizeFull();
        setPadding(false);
        setSpacing(false);
        addClassName("chat-view");

        // 创建头部
        Component header = createHeader();
        
        // 创建模式切换 Tabs
        Component modeSelector = createModeSelector();
        
        // 创建聊天容器（使用 Div + VerticalLayout 实现滚动）
        this.chatContainer = new VerticalLayout();
        chatContainer.setWidthFull();
        chatContainer.setPadding(true);
        chatContainer.setSpacing(true);
        chatContainer.addClassName("chat-container");
        
        // 创建滚动容器（使用 Div 配合 CSS）
        this.chatScroller = new Div();
        chatScroller.add(chatContainer);
        chatScroller.setSizeFull();
        chatScroller.getStyle().set("overflow-y", "auto");
        chatScroller.getStyle().set("overflow-x", "hidden");
        chatScroller.addClassName("chat-scroller");

        // 创建输入框
        this.messageInput = new MessageInput();
        messageInput.addSubmitListener(this::handleMessageSubmit);

        // 创建加载指示器
        this.loadingIndicator = new ProgressBar();
        loadingIndicator.setIndeterminate(true);
        loadingIndicator.setVisible(false);
        loadingIndicator.setWidthFull();

        // 添加欢迎消息
        addWelcomeMessage();

        // 布局组件
        VerticalLayout contentLayout = new VerticalLayout(chatScroller, loadingIndicator, messageInput);
        contentLayout.setSizeFull();
        contentLayout.setPadding(false);
        contentLayout.setSpacing(false);
        contentLayout.setFlexGrow(1, chatScroller);

        add(header, modeSelector, contentLayout);
        setFlexGrow(1, contentLayout);
    }

    /**
     * 创建模式选择器
     */
    private Component createModeSelector() {
        HorizontalLayout modeLayout = new HorizontalLayout();
        modeLayout.setPadding(true);
        modeLayout.setSpacing(true);
        modeLayout.setWidthFull();
        modeLayout.setAlignItems(Alignment.CENTER);
        modeLayout.addClassName("mode-selector");
        
        generalChatTab = new Tab("General Chat");
        bookAssistantTab = new Tab("Book Assistant");
        
        modeTabs = new Tabs(generalChatTab, bookAssistantTab);
        modeTabs.setSelectedTab(generalChatTab);
        modeTabs.addSelectedChangeListener(e -> {
            boolean previousMode = isBookAssistantMode;
            isBookAssistantMode = e.getSelectedTab() == bookAssistantTab;
            
            // 如果从通用聊天切换到书本助手，清理对话记忆
            // 如果从书本助手切换到通用聊天，也清理记忆（确保每个模式有独立的对话上下文）
            if (previousMode != isBookAssistantMode) {
                chatService.clearMemory();
                System.out.println("模式切换: " + (isBookAssistantMode ? "Book Assistant" : "General Chat") + " - 已清理对话记忆");
            }
            
            logModeChange();
        });
        
        modeLayout.add(modeTabs);
        modeLayout.setFlexGrow(1, modeTabs);
        
        return modeLayout;
    }
    
    /**
     * 记录模式切换日志
     */
    private void logModeChange() {
        System.out.println("模式切换: " + (isBookAssistantMode ? "Book Assistant" : "General Chat"));
    }

    /**
     * 创建头部组件
     */
    private Component createHeader() {
        VerticalLayout header = new VerticalLayout();
        header.setPadding(true);
        header.setSpacing(false);
        header.addClassName("chat-header");
        header.setWidthFull();

        H2 title = new H2("Java AI Integration Lab");
        title.addClassName(LumoUtility.Margin.NONE);
        title.addClassName(LumoUtility.FontSize.LARGE);

        Paragraph subtitle = new Paragraph("基于 Spring AI 的智能对话系统");
        subtitle.addClassName(LumoUtility.Margin.NONE);
        subtitle.addClassName(LumoUtility.TextColor.SECONDARY);
        subtitle.addClassName(LumoUtility.FontSize.SMALL);

        header.add(title, subtitle);
        return header;
    }

    /**
     * 添加欢迎消息
     */
    private void addWelcomeMessage() {
        addMessage("👋 你好！我是 AI 助手，有什么可以帮助你的吗？", false, List.of());
    }
    
    /**
     * 添加消息到聊天容器
     * 
     * @param text 消息文本
     * @param isUser 是否为用户消息
     * @param sources 检索到的文档片段（仅用于 AI 消息）
     */
    private void addMessage(String text, boolean isUser, List<DocumentService.SearchResult> sources) {
        VerticalLayout messageBubble = new VerticalLayout();
        messageBubble.setPadding(true);
        messageBubble.setSpacing(true);
        messageBubble.setWidthFull();
        messageBubble.addClassName("message-bubble");
        
        if (isUser) {
            // 用户消息：蓝色背景，白色文字，右对齐
            messageBubble.addClassName("user-message");
            messageBubble.addClassName(LumoUtility.Background.PRIMARY);
            messageBubble.addClassName(LumoUtility.TextColor.PRIMARY_CONTRAST);
            messageBubble.addClassName(LumoUtility.BorderRadius.MEDIUM);
            messageBubble.getStyle().set("align-self", "flex-end");
            messageBubble.getStyle().set("max-width", "70%");
            messageBubble.getStyle().set("margin-left", "auto");
            
            Span userText = new Span(text);
            userText.addClassName(LumoUtility.FontSize.MEDIUM);
            messageBubble.add(userText);
        } else {
            // AI 消息：浅灰色背景，左对齐
            messageBubble.addClassName("ai-message");
            messageBubble.addClassName(LumoUtility.Background.CONTRAST_5);
            messageBubble.addClassName(LumoUtility.BorderRadius.MEDIUM);
            messageBubble.getStyle().set("align-self", "flex-start");
            messageBubble.getStyle().set("max-width", "80%");
            messageBubble.getStyle().set("border-radius", "10px");
            
            // 将 Markdown 转换为 HTML
            Node document = markdownParser.parse(text);
            String html = htmlRenderer.render(document);
            
            // 创建 Div 组件并设置 innerHTML
            Div htmlDiv = new Div();
            htmlDiv.getElement().setProperty("innerHTML", "<div class='markdown-content'>" + html + "</div>");
            htmlDiv.getStyle().set("width", "100%");
            htmlDiv.addClassName("markdown-wrapper");
            
            // 添加 Markdown 样式（确保代码块使用等宽字体）
            htmlDiv.getElement().executeJs(
                "this.querySelectorAll('pre code, code').forEach(function(el) {" +
                "  el.style.fontFamily = 'monospace';" +
                "  el.style.fontSize = '0.9em';" +
                "  el.style.backgroundColor = 'var(--lumo-contrast-10pct)';" +
                "  el.style.padding = '2px 4px';" +
                "  el.style.borderRadius = '3px';" +
                "});" +
                "this.querySelectorAll('pre').forEach(function(el) {" +
                "  el.style.backgroundColor = 'var(--lumo-contrast-10pct)';" +
                "  el.style.padding = '12px';" +
                "  el.style.borderRadius = '4px';" +
                "  el.style.overflowX = 'auto';" +
                "});"
            );
            
            messageBubble.add(htmlDiv);
            
            // 如果有检索结果，添加 Sources 部分
            if (sources != null && !sources.isEmpty()) {
                Details sourcesDetails = createSourcesDetails(sources);
                messageBubble.add(sourcesDetails);
            }
        }
        
        chatContainer.add(messageBubble);
        
        // 滚动到底部
        scrollToBottom();
    }
    
    /**
     * 创建 Sources 详情组件
     * 
     * @param sources 检索结果列表
     * @return Details 组件
     */
    private Details createSourcesDetails(List<DocumentService.SearchResult> sources) {
        Details details = new Details();
        details.setSummaryText("Sources used (" + sources.size() + ")");
        details.addThemeVariants(DetailsVariant.FILLED);
        details.addClassName(LumoUtility.Margin.Top.SMALL);
        
        VerticalLayout sourcesContent = new VerticalLayout();
        sourcesContent.setPadding(false);
        sourcesContent.setSpacing(true);
        
        for (int i = 0; i < sources.size(); i++) {
            DocumentService.SearchResult result = sources.get(i);
            TextSegment segment = result.getSegment();
            String text = segment.text();
            
            // 限制预览长度
            String preview = text.length() > 200 ? text.substring(0, 200) + "..." : text;
            
            Div sourceItem = new Div();
            sourceItem.addClassName(LumoUtility.Padding.SMALL);
            sourceItem.addClassName(LumoUtility.Background.CONTRAST_10);
            sourceItem.addClassName(LumoUtility.BorderRadius.SMALL);
            
            // 构建源标识（包含元数据信息）
            String sourceLabel = buildSourceLabel(segment, i + 1);
            
            Span sourceLabelSpan = new Span(sourceLabel);
            sourceLabelSpan.addClassName(LumoUtility.FontWeight.BOLD);
            sourceLabelSpan.addClassName(LumoUtility.FontSize.SMALL);
            sourceLabelSpan.addClassName(LumoUtility.TextColor.PRIMARY);
            sourceLabelSpan.addClassName(LumoUtility.Margin.Bottom.XSMALL);
            
            Span sourceText = new Span(preview);
            sourceText.addClassName(LumoUtility.FontSize.SMALL);
            sourceText.getStyle().set("display", "block");
            sourceText.getStyle().set("margin-top", "4px");
            
            Span sourceScore = new Span(String.format(" (相似度: %.4f)", result.getScore()));
            sourceScore.addClassName(LumoUtility.TextColor.SECONDARY);
            sourceScore.addClassName(LumoUtility.FontSize.SMALL);
            sourceScore.getStyle().set("display", "block");
            sourceScore.getStyle().set("margin-top", "4px");
            
            sourceItem.add(sourceLabelSpan, sourceText, sourceScore);
            sourcesContent.add(sourceItem);
        }
        
        // 使用 add 方法添加内容（Details 的新 API）
        details.add(sourcesContent);
        return details;
    }
    
    /**
     * 构建源标签，包含元数据信息（Item ID, Chapter ID 等）
     * 
     * @param segment 文本片段
     * @param index 索引
     * @return 源标签字符串
     */
    private String buildSourceLabel(dev.langchain4j.data.segment.TextSegment segment, int index) {
        if (segment.metadata() == null || segment.metadata().asMap().isEmpty()) {
            return String.format("Source %d", index);
        }
        
        // 优先显示 Item ID（最常见）
        String itemId = segment.metadata().get("item_id");
        String itemLabel = segment.metadata().get("item_label");
        if (itemId != null && itemLabel != null) {
            return String.format("Source %d: %s", index, itemLabel);
        }
        
        // 其次显示 Chapter ID
        String chapterId = segment.metadata().get("chapter_id");
        String chapterLabel = segment.metadata().get("chapter_label");
        if (chapterId != null && chapterLabel != null) {
            return String.format("Source %d: %s", index, chapterLabel);
        }
        
        // 最后显示 Section ID
        String sectionId = segment.metadata().get("section_id");
        String sectionLabel = segment.metadata().get("section_label");
        if (sectionId != null && sectionLabel != null) {
            return String.format("Source %d: %s", index, sectionLabel);
        }
        
        // 如果没有结构化元数据，返回默认标签
        return String.format("Source %d", index);
    }

    /**
     * 处理消息提交
     */
    private void handleMessageSubmit(MessageInput.SubmitEvent event) {
        String userMessage = event.getValue();
        if (userMessage == null || userMessage.trim().isEmpty()) {
            return;
        }

        // 禁用输入框
        messageInput.setEnabled(false);
        loadingIndicator.setVisible(true);

        // 添加用户消息
        addMessage(userMessage, true, List.of());

        // 创建 AI 消息占位符
        VerticalLayout aiMessagePlaceholder = new VerticalLayout();
        aiMessagePlaceholder.setPadding(true);
        aiMessagePlaceholder.addClassName("ai-message");
        aiMessagePlaceholder.addClassName(LumoUtility.Background.CONTRAST_5);
        aiMessagePlaceholder.addClassName(LumoUtility.BorderRadius.MEDIUM);
        aiMessagePlaceholder.getStyle().set("align-self", "flex-start");
        aiMessagePlaceholder.getStyle().set("max-width", "80%");
        aiMessagePlaceholder.getStyle().set("border-radius", "10px");
        
        Span placeholderText = new Span("正在思考...");
        placeholderText.addClassName(LumoUtility.TextColor.SECONDARY);
        aiMessagePlaceholder.add(placeholderText);
        
        chatContainer.add(aiMessagePlaceholder);
        currentAiMessageComponent.set(aiMessagePlaceholder);
        currentAiResponse.set("");
        
        // 滚动到底部
        scrollToBottom();

        // 根据模式选择不同的服务方法
        if (isBookAssistantMode) {
            // RAG 模式：使用文档检索增强生成
            handleRagMessage(userMessage);
        } else {
            // 普通模式：直接调用 AI
            handleGeneralMessage(userMessage);
        }
    }
    
    /**
     * 处理普通聊天消息
     */
    private void handleGeneralMessage(String userMessage) {
        // 调用 ChatService 获取流式响应
        Flux<String> responseStream = chatService.streamResponse(userMessage);
        
        // 订阅流式响应
        subscribeToResponseStream(responseStream, null);
    }
    
    /**
     * 处理 RAG 增强的聊天消息
     */
    private void handleRagMessage(String userMessage) {
        // 调用 ChatService 获取 RAG 增强的流式响应
        ChatService.RagResponse ragResponse = chatService.streamRagResponse(userMessage);
        Flux<String> responseStream = ragResponse.getResponseStream();
        Mono<List<DocumentService.SearchResult>> searchResultsMono = ragResponse.getSearchResultsMono();
        
        // 异步获取检索结果，不阻塞 UI
        searchResultsMono.subscribe(
            results -> {
                currentSearchResults.set(results);
            },
            error -> {
                System.err.println("获取检索结果失败: " + error.getMessage());
                currentSearchResults.set(List.of());
            }
        );
        
        // 订阅流式响应
        subscribeToResponseStream(responseStream, searchResultsMono);
    }
    
    /**
     * 订阅响应流（通用方法）
     */
    private void subscribeToResponseStream(Flux<String> responseStream, Mono<List<DocumentService.SearchResult>> searchResultsMono) {
        responseStream.subscribe(
                chunk -> {
                    // 累积响应内容
                    String currentText = currentAiResponse.get();
                    String newText = currentText + chunk;
                    currentAiResponse.set(newText);
                    
                    // 标记有待更新的内容，使用节流减少更新频率
                    if (!pendingUpdate.getAndSet(true)) {
                        // 使用节流：每 100ms 最多更新一次 UI
                        updateScheduler.schedule(() -> {
                            if (pendingUpdate.getAndSet(false)) {
                                updateMessageUI();
                            }
                        }, 100, TimeUnit.MILLISECONDS);
                    }
                },
                error -> {
                    // 处理错误
                    getUI().ifPresent(ui -> ui.access(() -> {
                        Component currentComponent = currentAiMessageComponent.get();
                        if (currentComponent != null) {
                            chatContainer.remove(currentComponent);
                            addMessage("❌ 发生错误：" + error.getMessage(), false, List.of());
                        }
                        loadingIndicator.setVisible(false);
                        messageInput.setEnabled(true);
                    }));
                },
                () -> {
                    // 完成 - 确保最终内容被更新
                    String finalText = currentAiResponse.get();
                    System.out.println("流式响应完成，最终内容长度: " + (finalText != null ? finalText.length() : 0));
                    
                    // 立即更新最终内容
                    updateMessageUI();
                    
                    // 如果是 RAG 模式，显示 Sources
                    if (isBookAssistantMode && searchResultsMono != null) {
                        searchResultsMono.subscribe(
                            results -> {
                                getUI().ifPresent(ui -> ui.access(() -> {
                                    // Sources 已经在 addMessage 中处理
                                    updateAiMessageWithSources(results);
                                }));
                            },
                            error -> {
                                System.err.println("获取检索结果失败: " + error.getMessage());
                            }
                        );
                    }
                    
                    getUI().ifPresent(ui -> ui.access(() -> {
                        loadingIndicator.setVisible(false);
                        messageInput.setEnabled(true);
                        messageInput.focus();
                    }));
                }
        );
    }
    
    /**
     * 更新 AI 消息，添加 Sources 信息
     */
    private void updateAiMessageWithSources(List<DocumentService.SearchResult> searchResults) {
        Component currentComponent = currentAiMessageComponent.get();
        if (currentComponent == null || searchResults == null || searchResults.isEmpty()) {
            return;
        }
        
        // 如果当前组件是 VerticalLayout，添加 Sources Details
        if (currentComponent instanceof VerticalLayout) {
            VerticalLayout messageLayout = (VerticalLayout) currentComponent;
            // 检查是否已经添加了 Sources（避免重复添加）
            boolean hasSources = messageLayout.getChildren()
                    .anyMatch(child -> child instanceof Details);
            
            if (!hasSources) {
                Details sourcesDetails = createSourcesDetails(searchResults);
                messageLayout.add(sourcesDetails);
            }
        }
    }
    
    /**
     * 更新消息 UI（线程安全，使用节流优化性能）
     */
    private void updateMessageUI() {
        getUI().ifPresent(ui -> ui.access(() -> {
            Component currentComponent = currentAiMessageComponent.get();
            String currentText = currentAiResponse.get();
            
            if (currentComponent != null && currentText != null && !currentText.isEmpty()) {
                // 更新现有消息组件的内容
                if (currentComponent instanceof VerticalLayout) {
                    VerticalLayout messageLayout = (VerticalLayout) currentComponent;
                    
                    // 移除旧的文本组件（保留 Sources Details 如果存在）
                    messageLayout.removeAll();
                    
                    // 将 Markdown 转换为 HTML
                    Node document = markdownParser.parse(currentText);
                    String html = htmlRenderer.render(document);
                    
                    // 创建 Div 组件并设置 innerHTML
                    Div htmlDiv = new Div();
                    htmlDiv.getElement().setProperty("innerHTML", "<div class='markdown-content'>" + html + "</div>");
                    htmlDiv.getStyle().set("width", "100%");
                    htmlDiv.addClassName("markdown-wrapper");
                    
                    // 添加 Markdown 样式
                    htmlDiv.getElement().executeJs(
                        "this.querySelectorAll('pre code, code').forEach(function(el) {" +
                        "  el.style.fontFamily = 'monospace';" +
                        "  el.style.fontSize = '0.9em';" +
                        "  el.style.backgroundColor = 'var(--lumo-contrast-10pct)';" +
                        "  el.style.padding = '2px 4px';" +
                        "  el.style.borderRadius = '3px';" +
                        "});" +
                        "this.querySelectorAll('pre').forEach(function(el) {" +
                        "  el.style.backgroundColor = 'var(--lumo-contrast-10pct)';" +
                        "  el.style.padding = '12px';" +
                        "  el.style.borderRadius = '4px';" +
                        "  el.style.overflowX = 'auto';" +
                        "});"
                    );
                    
                    messageLayout.add(htmlDiv);
                    
                    // 如果有检索结果，添加 Sources
                    List<DocumentService.SearchResult> sources = currentSearchResults.get();
                    if (sources != null && !sources.isEmpty()) {
                        Details sourcesDetails = createSourcesDetails(sources);
                        messageLayout.add(sourcesDetails);
                    }
                    
                    // 滚动到底部
                    scrollToBottom();
                }
            }
        }));
    }
    
    /**
     * 滚动到底部
     */
    private void scrollToBottom() {
        chatScroller.getElement().executeJs(
            "setTimeout(function() { this.scrollTop = this.scrollHeight; }, 100);"
        );
    }
    
    /**
     * 清理资源
     */
    @PreDestroy
    public void destroy() {
        if (updateScheduler != null && !updateScheduler.isShutdown()) {
            updateScheduler.shutdown();
            try {
                if (!updateScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                    updateScheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                updateScheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}

