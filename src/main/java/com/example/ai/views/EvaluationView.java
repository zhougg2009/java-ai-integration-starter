package com.example.ai.views;

import com.example.ai.config.RagProperties;
import com.example.ai.service.RagEvaluationService;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.checkbox.Checkbox;
import com.vaadin.flow.component.details.Details;
import com.vaadin.flow.component.details.DetailsVariant;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.notification.NotificationVariant;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.progressbar.ProgressBar;
import com.vaadin.flow.component.textfield.IntegerField;
import com.vaadin.flow.data.renderer.ComponentRenderer;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;
import com.vaadin.flow.theme.lumo.LumoUtility;
import org.commonmark.node.Node;
import org.commonmark.parser.Parser;
import org.commonmark.renderer.html.HtmlRenderer;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.List;

/**
 * RAG 性能评估中心
 * 提供实时性能跟踪和交互式评估仪表板
 */
@Route("evaluation")
@PageTitle("RAG Performance Evaluation Center")
public class EvaluationView extends VerticalLayout {

    private final RagEvaluationService evaluationService;
    private final RagProperties ragProperties;
    
    // UI 组件
    private Button generateTestSetButton;
    private Button startBatchEvaluationButton;
    private IntegerField numQuestionsField;
    private Checkbox hydeCheckbox;
    private Checkbox stepbackCheckbox;
    private Checkbox rerankCheckbox;
    private Checkbox hybridSearchCheckbox;
    private ProgressBar progressBar;
    private Span statusLabel;
    private Grid<RagEvaluationService.EvaluationResult> resultsGrid;
    private Details reportDetails;
    private Div summaryCard;
    
    // Markdown 处理器
    private final Parser markdownParser;
    private final HtmlRenderer htmlRenderer;
    
    public EvaluationView(RagEvaluationService evaluationService, RagProperties ragProperties) {
        this.evaluationService = evaluationService;
        this.ragProperties = ragProperties;
        
        // 初始化 Markdown 处理器
        this.markdownParser = Parser.builder().build();
        this.htmlRenderer = HtmlRenderer.builder().build();
        
        setSizeFull();
        setPadding(true);
        setSpacing(true);
        addClassName("evaluation-view");
        
        // 创建标题
        H2 title = new H2("RAG Performance Evaluation Center");
        title.addClassName(LumoUtility.FontSize.XXXLARGE);
        title.addClassName(LumoUtility.FontWeight.BOLD);
        title.addClassName(LumoUtility.Margin.Bottom.LARGE);
        
        // 创建控制区域
        VerticalLayout controlArea = createControlArea();
        
        // 创建状态区域
        VerticalLayout statusArea = createStatusArea();
        
        // 创建摘要卡片
        summaryCard = createSummaryCard();
        summaryCard.setVisible(false);
        
        // 创建结果区域
        VerticalLayout resultsArea = createResultsArea();
        
        // 布局
        add(title, controlArea, statusArea, summaryCard, resultsArea);
    }
    
    /**
     * 创建控制区域
     */
    private VerticalLayout createControlArea() {
        VerticalLayout controlArea = new VerticalLayout();
        controlArea.setWidthFull();
        controlArea.setPadding(true);
        controlArea.getStyle().set("background-color", "var(--lumo-contrast-5pct)");
        controlArea.getStyle().set("border-radius", "8px");
        controlArea.setSpacing(true);
        controlArea.addClassName(LumoUtility.Margin.Bottom.MEDIUM);
        
        H3 sectionTitle = new H3("Control Panel");
        sectionTitle.addClassName(LumoUtility.FontSize.LARGE);
        sectionTitle.addClassName(LumoUtility.FontWeight.SEMIBOLD);
        sectionTitle.addClassName(LumoUtility.Margin.Bottom.SMALL);
        
        // 测试集生成区域
        HorizontalLayout testSetLayout = new HorizontalLayout();
        testSetLayout.setAlignItems(Alignment.END);
        testSetLayout.setSpacing(true);
        
        numQuestionsField = new IntegerField("Number of Questions");
        numQuestionsField.setValue(10);
        numQuestionsField.setMin(1);
        numQuestionsField.setMax(100);
        numQuestionsField.setHelperText("1-100");
        numQuestionsField.setWidth("200px");
        
        generateTestSetButton = new Button("Generate Test Set", e -> generateTestSet());
        generateTestSetButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        
        testSetLayout.add(numQuestionsField, generateTestSetButton);
        
        // 批量评估按钮
        startBatchEvaluationButton = new Button("Start Batch Evaluation", e -> startBatchEvaluation());
        startBatchEvaluationButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY, ButtonVariant.LUMO_SUCCESS);
        startBatchEvaluationButton.addClassName(LumoUtility.Margin.Top.SMALL);
        
        // RAG 功能开关区域
        H3 featuresTitle = new H3("RAG Features Toggle");
        featuresTitle.addClassName(LumoUtility.FontSize.MEDIUM);
        featuresTitle.addClassName(LumoUtility.FontWeight.SEMIBOLD);
        featuresTitle.addClassName(LumoUtility.Margin.Top.MEDIUM);
        featuresTitle.addClassName(LumoUtility.Margin.Bottom.SMALL);
        
        VerticalLayout featuresLayout = new VerticalLayout();
        featuresLayout.setSpacing(true);
        featuresLayout.setPadding(false);
        
        hydeCheckbox = new Checkbox("HyDE (Hypothetical Document Embeddings)", ragProperties.isHydeEnabled());
        hydeCheckbox.addValueChangeListener(e -> {
            ragProperties.setHydeEnabled(e.getValue());
            logFeatureChange("HyDE", e.getValue());
        });
        
        stepbackCheckbox = new Checkbox("Step-back Prompting", ragProperties.isStepbackEnabled());
        stepbackCheckbox.addValueChangeListener(e -> {
            ragProperties.setStepbackEnabled(e.getValue());
            logFeatureChange("Step-back", e.getValue());
        });
        
        rerankCheckbox = new Checkbox("Reranking", ragProperties.isRerankEnabled());
        rerankCheckbox.addValueChangeListener(e -> {
            ragProperties.setRerankEnabled(e.getValue());
            logFeatureChange("Reranking", e.getValue());
        });
        
        hybridSearchCheckbox = new Checkbox("Hybrid Search (Vector + Keyword)", ragProperties.isHybridSearchEnabled());
        hybridSearchCheckbox.addValueChangeListener(e -> {
            ragProperties.setHybridSearchEnabled(e.getValue());
            logFeatureChange("Hybrid Search", e.getValue());
        });
        
        HorizontalLayout featuresRow1 = new HorizontalLayout(hydeCheckbox, stepbackCheckbox);
        HorizontalLayout featuresRow2 = new HorizontalLayout(rerankCheckbox, hybridSearchCheckbox);
        featuresRow1.setSpacing(true);
        featuresRow2.setSpacing(true);
        
        featuresLayout.add(featuresRow1, featuresRow2);
        
        controlArea.add(sectionTitle, testSetLayout, startBatchEvaluationButton, featuresTitle, featuresLayout);
        
        return controlArea;
    }
    
    /**
     * 创建状态区域
     */
    private VerticalLayout createStatusArea() {
        VerticalLayout statusArea = new VerticalLayout();
        statusArea.setWidthFull();
        statusArea.setSpacing(true);
        statusArea.addClassName(LumoUtility.Margin.Bottom.MEDIUM);
        
        // 进度条
        progressBar = new ProgressBar();
        progressBar.setIndeterminate(false);
        progressBar.setValue(0.0);
        progressBar.setVisible(false);
        progressBar.setWidthFull();
        
        // 状态标签
        statusLabel = new Span("Current Status: Idle");
        statusLabel.addClassName(LumoUtility.FontSize.MEDIUM);
        statusLabel.addClassName(LumoUtility.FontWeight.SEMIBOLD);
        
        statusArea.add(progressBar, statusLabel);
        
        return statusArea;
    }
    
    /**
     * 创建摘要卡片
     */
    private Div createSummaryCard() {
        Div card = new Div();
        card.setWidthFull();
        card.getStyle().set("padding", "16px");
        card.getStyle().set("background-color", "var(--lumo-primary-color-10pct)");
        card.getStyle().set("border-radius", "8px");
        card.getStyle().set("border", "2px solid var(--lumo-primary-color)");
        card.addClassName(LumoUtility.Margin.Bottom.MEDIUM);
        
        H3 cardTitle = new H3("Evaluation Summary");
        cardTitle.addClassName(LumoUtility.FontSize.LARGE);
        cardTitle.addClassName(LumoUtility.FontWeight.BOLD);
        cardTitle.addClassName(LumoUtility.Margin.Bottom.MEDIUM);
        
        Div metricsContainer = new Div();
        metricsContainer.getStyle().set("display", "grid");
        metricsContainer.getStyle().set("grid-template-columns", "repeat(auto-fit, minmax(200px, 1fr))");
        metricsContainer.getStyle().set("gap", "16px");
        metricsContainer.getStyle().set("padding", "0");
        
        // 这些将在评估完成后更新
        Span avgFaithfulness = new Span();
        avgFaithfulness.setId("avg-faithfulness");
        avgFaithfulness.addClassName(LumoUtility.FontSize.LARGE);
        avgFaithfulness.addClassName(LumoUtility.FontWeight.BOLD);
        
        Span avgRelevance = new Span();
        avgRelevance.setId("avg-relevance");
        avgRelevance.addClassName(LumoUtility.FontSize.LARGE);
        avgRelevance.addClassName(LumoUtility.FontWeight.BOLD);
        
        Span avgContextPrecision = new Span();
        avgContextPrecision.setId("avg-context-precision");
        avgContextPrecision.addClassName(LumoUtility.FontSize.LARGE);
        avgContextPrecision.addClassName(LumoUtility.FontWeight.BOLD);
        
        Span avgAnswerSimilarity = new Span();
        avgAnswerSimilarity.setId("avg-answer-similarity");
        avgAnswerSimilarity.addClassName(LumoUtility.FontSize.LARGE);
        avgAnswerSimilarity.addClassName(LumoUtility.FontWeight.BOLD);
        
        metricsContainer.add(
            createMetricItem("Avg Faithfulness", avgFaithfulness),
            createMetricItem("Avg Relevance", avgRelevance),
            createMetricItem("Avg Context Precision", avgContextPrecision),
            createMetricItem("Avg Answer Similarity", avgAnswerSimilarity)
        );
        
        card.add(cardTitle, metricsContainer);
        
        return card;
    }
    
    /**
     * 创建指标项
     */
    private Div createMetricItem(String label, Span value) {
        Div item = new Div();
        item.getStyle().set("text-align", "center");
        item.getStyle().set("padding", "12px");
        item.getStyle().set("background-color", "white");
        item.getStyle().set("border-radius", "4px");
        
        Span labelSpan = new Span(label);
        labelSpan.addClassName(LumoUtility.FontSize.SMALL);
        labelSpan.addClassName(LumoUtility.TextColor.SECONDARY);
        labelSpan.addClassName(LumoUtility.Margin.Bottom.SMALL);
        labelSpan.getStyle().set("display", "block");
        
        value.getStyle().set("display", "block");
        value.getStyle().set("color", "var(--lumo-primary-color)");
        
        item.add(labelSpan, value);
        return item;
    }
    
    /**
     * 创建结果区域
     */
    private VerticalLayout createResultsArea() {
        VerticalLayout resultsArea = new VerticalLayout();
        resultsArea.setWidthFull();
        resultsArea.setSpacing(true);
        resultsArea.setFlexGrow(1);
        
        // 结果表格
        H3 resultsTitle = new H3("Evaluation Results");
        resultsTitle.addClassName(LumoUtility.FontSize.LARGE);
        resultsTitle.addClassName(LumoUtility.FontWeight.SEMIBOLD);
        
        resultsGrid = createResultsGrid();
        resultsGrid.setVisible(false);
        resultsGrid.setHeight("400px");
        
        // 报告详情
        reportDetails = new Details("Evaluation Report", new Div());
        reportDetails.setOpened(false);
        reportDetails.addThemeVariants(DetailsVariant.FILLED);
        reportDetails.setVisible(false);
        reportDetails.setWidthFull();
        
        resultsArea.add(resultsTitle, resultsGrid, reportDetails);
        resultsArea.setFlexGrow(1, resultsGrid);
        
        return resultsArea;
    }
    
    /**
     * 创建结果表格
     */
    private Grid<RagEvaluationService.EvaluationResult> createResultsGrid() {
        Grid<RagEvaluationService.EvaluationResult> grid = new Grid<>(RagEvaluationService.EvaluationResult.class, false);
        grid.setWidthFull();
        
        // 问题列
        grid.addColumn(RagEvaluationService.EvaluationResult::getQuestion)
                .setHeader("Question")
                .setFlexGrow(3)
                .setAutoWidth(true);
        
        // Faithfulness 分数列
        grid.addColumn(new ComponentRenderer<>(result -> {
            double score = result.getFaithfulnessScore();
            return createScoreSpan(score, "Faithfulness");
        })).setHeader("Faithfulness").setWidth("120px");
        
        // Relevance 分数列
        grid.addColumn(new ComponentRenderer<>(result -> {
            double score = result.getRelevanceScore();
            return createScoreSpan(score, "Relevance");
        })).setHeader("Relevance").setWidth("120px");
        
        // Context Precision 分数列
        grid.addColumn(new ComponentRenderer<>(result -> {
            double score = result.getContextPrecisionScore();
            return createScoreSpan(score, "Context Precision");
        })).setHeader("Context Precision").setWidth("150px");
        
        // Answer Similarity 分数列
        grid.addColumn(new ComponentRenderer<>(result -> {
            double score = result.getAnswerSimilarityScore();
            return createScoreSpan(score, "Answer Similarity");
        })).setHeader("Answer Similarity").setWidth("150px");
        
        return grid;
    }
    
    /**
     * 创建分数显示 Span
     */
    private Span createScoreSpan(double score, String label) {
        Span span = new Span(String.format("%.3f", score));
        if (score >= 0.8) {
            span.addClassName(LumoUtility.TextColor.SUCCESS);
        } else if (score >= 0.6) {
            span.addClassName(LumoUtility.TextColor.WARNING);
        } else {
            span.addClassName(LumoUtility.TextColor.ERROR);
        }
        span.getElement().setAttribute("title", label + ": " + String.format("%.3f", score));
        return span;
    }
    
    /**
     * 生成测试集
     */
    private void generateTestSet() {
        int numQuestions = numQuestionsField.getValue() != null ? numQuestionsField.getValue() : 10;
        
        generateTestSetButton.setEnabled(false);
        updateStatus("Generating Test Set...", true);
        
        Mono.fromCallable(() -> {
            return evaluationService.generateTestSet(numQuestions);
        })
        .subscribeOn(Schedulers.boundedElastic())
        .subscribe(
            testQuestions -> {
                getUI().ifPresent(ui -> ui.access(() -> {
                    generateTestSetButton.setEnabled(true);
                    updateStatus("Idle", false);
                    
                    Notification notification = Notification.show(
                        String.format("Test set generated successfully! %d questions created.", testQuestions.size()),
                        3000,
                        Notification.Position.TOP_END
                    );
                    notification.addThemeVariants(NotificationVariant.LUMO_SUCCESS);
                }));
            },
            error -> {
                getUI().ifPresent(ui -> ui.access(() -> {
                    generateTestSetButton.setEnabled(true);
                    updateStatus("Idle", false);
                    
                    Notification notification = Notification.show(
                        "Failed to generate test set: " + error.getMessage(),
                        5000,
                        Notification.Position.TOP_END
                    );
                    notification.addThemeVariants(NotificationVariant.LUMO_ERROR);
                }));
            }
        );
    }
    
    /**
     * 开始批量评估
     */
    private void startBatchEvaluation() {
        startBatchEvaluationButton.setEnabled(false);
        generateTestSetButton.setEnabled(false);
        updateStatus("Running Batch Evaluation...", true);
        resultsGrid.setVisible(false);
        summaryCard.setVisible(false);
        reportDetails.setVisible(false);
        
        Mono.fromCallable(() -> {
            return evaluationService.runFullEvaluation(
                numQuestionsField.getValue() != null ? numQuestionsField.getValue() : 10
            );
        })
        .subscribeOn(Schedulers.boundedElastic())
        .doOnNext(results -> {
            // 更新进度
            getUI().ifPresent(ui -> ui.access(() -> {
                updateStatus("Evaluation Complete!", false);
                displayResults(results);
            }));
        })
        .doOnError(error -> {
            getUI().ifPresent(ui -> ui.access(() -> {
                startBatchEvaluationButton.setEnabled(true);
                generateTestSetButton.setEnabled(true);
                updateStatus("Evaluation Failed: " + error.getMessage(), false);
                
                Notification notification = Notification.show(
                    "Evaluation failed: " + error.getMessage(),
                    5000,
                    Notification.Position.TOP_END
                );
                notification.addThemeVariants(NotificationVariant.LUMO_ERROR);
            }));
        })
        .subscribe();
    }
    
    /**
     * 显示评估结果
     */
    private void displayResults(List<RagEvaluationService.EvaluationResult> results) {
        if (results == null || results.isEmpty()) {
            return;
        }
        
        // 更新表格
        resultsGrid.setItems(results);
        resultsGrid.setVisible(true);
        
        // 计算并显示平均分数
        double avgFaithfulness = results.stream()
                .mapToDouble(RagEvaluationService.EvaluationResult::getFaithfulnessScore)
                .average()
                .orElse(0.0);
        
        double avgRelevance = results.stream()
                .mapToDouble(RagEvaluationService.EvaluationResult::getRelevanceScore)
                .average()
                .orElse(0.0);
        
        double avgContextPrecision = results.stream()
                .mapToDouble(RagEvaluationService.EvaluationResult::getContextPrecisionScore)
                .average()
                .orElse(0.0);
        
        double avgAnswerSimilarity = results.stream()
                .mapToDouble(RagEvaluationService.EvaluationResult::getAnswerSimilarityScore)
                .average()
                .orElse(0.0);
        
        // 更新摘要卡片
        updateSummaryCard(avgFaithfulness, avgRelevance, avgContextPrecision, avgAnswerSimilarity);
        summaryCard.setVisible(true);
        
        // 显示报告
        displayReport();
        
        // 重新启用按钮
        startBatchEvaluationButton.setEnabled(true);
        generateTestSetButton.setEnabled(true);
    }
    
    /**
     * 更新摘要卡片
     */
    private void updateSummaryCard(double avgFaithfulness, double avgRelevance, 
                                   double avgContextPrecision, double avgAnswerSimilarity) {
        getUI().ifPresent(ui -> {
            ui.access(() -> {
                // 直接更新摘要卡片内容
                Div metricsContainer = (Div) summaryCard.getChildren()
                    .filter(c -> c instanceof Div && c.getStyle().get("display").equals("grid"))
                    .findFirst()
                    .orElse(null);
                
                if (metricsContainer != null) {
                    metricsContainer.removeAll();
                    metricsContainer.add(
                        createMetricItem("Avg Faithfulness", createScoreSpan(avgFaithfulness, "Faithfulness")),
                        createMetricItem("Avg Relevance", createScoreSpan(avgRelevance, "Relevance")),
                        createMetricItem("Avg Context Precision", createScoreSpan(avgContextPrecision, "Context Precision")),
                        createMetricItem("Avg Answer Similarity", createScoreSpan(avgAnswerSimilarity, "Answer Similarity"))
                    );
                }
            });
        });
    }
    
    /**
     * 显示评估报告
     */
    private void displayReport() {
        try {
            java.nio.file.Path reportPath = java.nio.file.Paths.get("external-docs/evaluation_report.md");
            if (!java.nio.file.Files.exists(reportPath)) {
                reportDetails.setVisible(false);
                return;
            }
            
            String reportContent = java.nio.file.Files.readString(reportPath);
            
            // 将 Markdown 转换为 HTML
            Node document = markdownParser.parse(reportContent);
            String html = htmlRenderer.render(document);
            
            // 创建报告内容 Div
            Div reportContentDiv = new Div();
            reportContentDiv.getElement().setProperty("innerHTML", html);
            reportContentDiv.getStyle().set("padding", "16px");
            reportContentDiv.getStyle().set("max-height", "600px");
            reportContentDiv.getStyle().set("overflow", "auto");
            
            reportDetails.removeAll();
            reportDetails.add(reportContentDiv);
            reportDetails.setVisible(true);
            
        } catch (Exception e) {
            // 如果读取失败，隐藏报告区域
            reportDetails.setVisible(false);
        }
    }
    
    /**
     * 更新状态
     */
    private void updateStatus(String status, boolean showProgress) {
        statusLabel.setText("Current Status: " + status);
        progressBar.setVisible(showProgress);
        if (showProgress) {
            progressBar.setIndeterminate(true);
        } else {
            progressBar.setIndeterminate(false);
            progressBar.setValue(0.0);
        }
    }
    
    /**
     * 记录功能开关变化
     */
    private void logFeatureChange(String featureName, boolean enabled) {
        // 可以添加日志记录
        System.out.println("Feature " + featureName + " " + (enabled ? "enabled" : "disabled"));
    }
}
