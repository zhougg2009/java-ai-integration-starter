package com.example.ai.views;

import com.example.ai.service.ChatService;
import com.example.ai.service.DocumentService;
import com.vaadin.flow.component.Component;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.html.Paragraph;
import com.vaadin.flow.component.messages.MessageInput;
import com.vaadin.flow.component.messages.MessageList;
import com.vaadin.flow.component.messages.MessageListItem;
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
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.ArrayList;
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
    private final MessageList messageList;
    private final MessageInput messageInput;
    private final List<MessageListItem> messages;
    private final AtomicReference<String> currentAiResponse;
    private final AtomicReference<MessageListItem> currentAiMessageItem;
    private final AtomicReference<Integer> currentAiMessageIndex;
    private final AtomicReference<Boolean> pendingUpdate;
    private final ScheduledExecutorService updateScheduler;
    private ProgressBar loadingIndicator;
    
    // 模式切换
    private Tabs modeTabs;
    private Tab generalChatTab;
    private Tab bookAssistantTab;
    private boolean isBookAssistantMode = false;
    
    // RAG 模式下的检索结果存储
    private final AtomicReference<List<DocumentService.SearchResult>> currentSearchResults;

    public ChatView(ChatService chatService) {
        this.chatService = chatService;
        this.messages = new ArrayList<>();
        this.currentAiResponse = new AtomicReference<>("");
        this.currentAiMessageItem = new AtomicReference<>();
        this.currentAiMessageIndex = new AtomicReference<>(-1);
        this.pendingUpdate = new AtomicReference<>(false);
        this.updateScheduler = Executors.newSingleThreadScheduledExecutor();
        this.currentSearchResults = new AtomicReference<>(List.of());

        setSizeFull();
        setPadding(false);
        setSpacing(false);
        addClassName("chat-view");

        // 创建头部
        Component header = createHeader();
        
        // 创建模式切换 Tabs
        Component modeSelector = createModeSelector();
        
        // 创建消息列表
        this.messageList = new MessageList();
        messageList.setSizeFull();

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
        VerticalLayout contentLayout = new VerticalLayout(messageList, loadingIndicator, messageInput);
        contentLayout.setSizeFull();
        contentLayout.setPadding(false);
        contentLayout.setSpacing(false);
        contentLayout.setFlexGrow(1, messageList);

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
            isBookAssistantMode = e.getSelectedTab() == bookAssistantTab;
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
        MessageListItem welcomeMessage = new MessageListItem(
                "👋 你好！我是 AI 助手，有什么可以帮助你的吗？",
                Instant.now(),
                "AI Assistant"
        );
        welcomeMessage.addThemeNames("ai-message");
        messages.add(welcomeMessage);
        messageList.setItems(messages);
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
        MessageListItem userMessageItem = new MessageListItem(
                userMessage,
                Instant.now(),
                "You"
        );
        userMessageItem.addThemeNames("user-message");
        messages.add(userMessageItem);
        messageList.setItems(messages);

        // 创建 AI 消息占位符
        MessageListItem aiMessageItem = new MessageListItem(
                "正在思考...",
                Instant.now(),
                "AI Assistant"
        );
        aiMessageItem.addThemeNames("ai-message");
        messages.add(aiMessageItem);
        int aiMessageIndex = messages.size() - 1;
        currentAiMessageItem.set(aiMessageItem);
        currentAiMessageIndex.set(aiMessageIndex);
        currentAiResponse.set("");
        messageList.setItems(new ArrayList<>(messages));

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
                // 在响应完成后显示 Sources
                getUI().ifPresent(ui -> ui.access(() -> {
                    // Sources 将在响应完成后显示
                }));
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
                        MessageListItem currentItem = currentAiMessageItem.get();
                        if (currentItem != null) {
                            currentItem.setText("❌ 发生错误：" + error.getMessage());
                            messageList.setItems(messages);
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
                                    addSourcesToMessage(results);
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
     * 在 AI 消息下方添加 Sources 信息
     */
    private void addSourcesToMessage(List<DocumentService.SearchResult> searchResults) {
        if (searchResults == null || searchResults.isEmpty()) {
            return;
        }
        
        Integer itemIndex = currentAiMessageIndex.get();
        if (itemIndex < 0 || itemIndex >= messages.size()) {
            return;
        }
        
        // 构建 Sources 文本
        StringBuilder sourcesText = new StringBuilder();
        sourcesText.append("\n\n---\n");
        sourcesText.append("**Sources used:**\n\n");
        
        for (int i = 0; i < searchResults.size(); i++) {
            DocumentService.SearchResult result = searchResults.get(i);
            TextSegment segment = result.getSegment();
            String text = segment.text();
            
            // 限制预览长度
            String preview = text.length() > 150 ? text.substring(0, 150) + "..." : text;
            
            sourcesText.append(String.format("%d. ", i + 1));
            sourcesText.append(preview);
            sourcesText.append(String.format(" (相似度: %.4f)", result.getScore()));
            sourcesText.append("\n\n");
        }
        
        // 更新消息内容，添加 Sources
        String currentText = currentAiResponse.get();
        String textWithSources = currentText + sourcesText.toString();
        
        MessageListItem currentItem = currentAiMessageItem.get();
        MessageListItem updatedItem = new MessageListItem(
            textWithSources,
            currentItem != null ? currentItem.getTime() : Instant.now(),
            currentItem != null ? currentItem.getUserName() : "AI Assistant"
        );
        updatedItem.addThemeNames("ai-message");
        messages.set(itemIndex, updatedItem);
        currentAiMessageItem.set(updatedItem);
        messageList.setItems(new ArrayList<>(messages));
    }
    
    /**
     * 更新消息 UI（线程安全，使用节流优化性能）
     */
    private void updateMessageUI() {
        getUI().ifPresent(ui -> ui.access(() -> {
            Integer itemIndex = currentAiMessageIndex.get();
            String currentText = currentAiResponse.get();
            MessageListItem currentItem = currentAiMessageItem.get();
            
            if (itemIndex >= 0 && itemIndex < messages.size() && currentText != null && !currentText.isEmpty()) {
                // 直接更新现有 MessageListItem 的文本，而不是创建新对象
                // 注意：MessageListItem 可能不支持直接 setText，所以我们需要替换
                MessageListItem updatedItem = new MessageListItem(
                        currentText,
                        currentItem != null ? currentItem.getTime() : Instant.now(),
                        currentItem != null ? currentItem.getUserName() : "AI Assistant"
                );
                updatedItem.addThemeNames("ai-message");
                messages.set(itemIndex, updatedItem);
                currentAiMessageItem.set(updatedItem);
                
                // 只在列表结构变化时才调用 setItems，这里直接更新单个项目
                // 使用 refreshItem 如果支持，否则使用 setItems
                messageList.setItems(new ArrayList<>(messages));
            }
        }));
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

