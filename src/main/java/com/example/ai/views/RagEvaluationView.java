package com.example.ai.views;

import com.example.ai.service.RagEvaluationService;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.html.Span;
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

/**
 * RAG 评估视图
 * 提供 UI 界面来运行评估、查看进度和结果
 */
@Route("rag-evaluation")
@PageTitle("RAG Evaluation")
public class RagEvaluationView extends VerticalLayout {

    private final RagEvaluationService evaluationService;
    
    // UI 组件
    private Button startEvaluationButton;
    private IntegerField numQuestionsField;
    private final ProgressBar progressBar;
    private final Span statusLabel;
    private final Grid<RagEvaluationService.EvaluationResult> resultsGrid;
    private final Div reportDiv;
    
    // Markdown 处理器
    private final Parser markdownParser;
    private final HtmlRenderer htmlRenderer;
    
    public RagEvaluationView(RagEvaluationService evaluationService) {
        this.evaluationService = evaluationService;
        
        // 初始化 Markdown 处理器
        this.markdownParser = Parser.builder().build();
        this.htmlRenderer = HtmlRenderer.builder().build();
        
        setSizeFull();
        setPadding(true);
        setSpacing(true);
        addClassName("rag-evaluation-view");
        
        // 创建标题
        H2 title = new H2("RAG 系统评估");
        title.addClassName(LumoUtility.FontSize.XXXLARGE);
        title.addClassName(LumoUtility.FontWeight.BOLD);
        title.addClassName(LumoUtility.Margin.Bottom.MEDIUM);
        
        // 创建控制面板
        VerticalLayout controlPanel = createControlPanel();
        
        // 创建进度条
        progressBar = new ProgressBar();
        progressBar.setIndeterminate(false);
        progressBar.setValue(0.0);
        progressBar.setVisible(false);
        progressBar.setWidthFull();
        progressBar.addClassName(LumoUtility.Margin.Bottom.MEDIUM);
        
        // 创建状态标签
        statusLabel = new Span();
        statusLabel.addClassName(LumoUtility.FontSize.SMALL);
        statusLabel.addClassName(LumoUtility.TextColor.SECONDARY);
        statusLabel.setVisible(false);
        
        // 创建结果表格
        resultsGrid = createResultsGrid();
        resultsGrid.setVisible(false);
        
        // 创建报告显示区域
        reportDiv = new Div();
        reportDiv.setWidthFull();
        reportDiv.getStyle().set("padding", "16px");
        reportDiv.getStyle().set("background-color", "var(--lumo-contrast-5pct)");
        reportDiv.getStyle().set("border-radius", "4px");
        reportDiv.getStyle().set("overflow", "auto");
        reportDiv.getStyle().set("max-height", "600px");
        reportDiv.setVisible(false);
        
        // 布局
        VerticalLayout contentLayout = new VerticalLayout();
        contentLayout.setWidthFull();
        contentLayout.setSpacing(true);
        contentLayout.add(controlPanel, progressBar, statusLabel, resultsGrid, reportDiv);
        
        add(title, contentLayout);
    }
    
    /**
     * 创建控制面板
     */
    private VerticalLayout createControlPanel() {
        VerticalLayout panel = new VerticalLayout();
        panel.setWidthFull();
        panel.setPadding(true);
        panel.getStyle().set("background-color", "var(--lumo-contrast-5pct)");
        panel.getStyle().set("border-radius", "4px");
        panel.setSpacing(true);
        
        // 问题数量输入
        numQuestionsField = new IntegerField("测试问题数量");
        numQuestionsField.setValue(10);
        numQuestionsField.setMin(1);
        numQuestionsField.setMax(100);
        numQuestionsField.setHelperText("输入要生成的测试问题数量（1-100）");
        numQuestionsField.setWidth("200px");
        
        // 开始评估按钮
        startEvaluationButton = new Button("开始完整评估", e -> startEvaluation());
        startEvaluationButton.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        startEvaluationButton.addClassName(LumoUtility.Margin.Top.SMALL);
        
        HorizontalLayout inputLayout = new HorizontalLayout();
        inputLayout.setAlignItems(Alignment.END);
        inputLayout.setSpacing(true);
        inputLayout.add(numQuestionsField, startEvaluationButton);
        
        Span description = new Span(
            "点击按钮将执行完整评估流程：\n" +
            "1. 生成测试集（从文档片段生成问题和答案对）\n" +
            "2. 批量测试（通过 RAG 管道运行每个问题）\n" +
            "3. 生成评估报告（包含 Faithfulness、Relevance、Context Precision、Answer Similarity 等指标）"
        );
        description.addClassName(LumoUtility.FontSize.SMALL);
        description.getStyle().set("white-space", "pre-line");
        description.addClassName(LumoUtility.TextColor.SECONDARY);
        
        panel.add(description, inputLayout);
        
        return panel;
    }
    
    /**
     * 创建结果表格
     */
    private Grid<RagEvaluationService.EvaluationResult> createResultsGrid() {
        Grid<RagEvaluationService.EvaluationResult> grid = new Grid<>(RagEvaluationService.EvaluationResult.class, false);
        grid.setWidthFull();
        grid.setHeight("400px");
        
        // 问题列
        grid.addColumn(RagEvaluationService.EvaluationResult::getQuestion)
                .setHeader("问题")
                .setFlexGrow(2)
                .setAutoWidth(true);
        
        // Faithfulness 分数列
        grid.addColumn(new ComponentRenderer<>(result -> {
            double score = result.getFaithfulnessScore();
            Span span = new Span(String.format("%.3f", score));
            if (score >= 0.8) {
                span.addClassName(LumoUtility.TextColor.SUCCESS);
            } else if (score >= 0.6) {
                span.addClassName(LumoUtility.TextColor.WARNING);
            } else {
                span.addClassName(LumoUtility.TextColor.ERROR);
            }
            return span;
        })).setHeader("Faithfulness").setWidth("120px");
        
        // Relevance 分数列
        grid.addColumn(new ComponentRenderer<>(result -> {
            double score = result.getRelevanceScore();
            Span span = new Span(String.format("%.3f", score));
            if (score >= 0.8) {
                span.addClassName(LumoUtility.TextColor.SUCCESS);
            } else if (score >= 0.6) {
                span.addClassName(LumoUtility.TextColor.WARNING);
            } else {
                span.addClassName(LumoUtility.TextColor.ERROR);
            }
            return span;
        })).setHeader("Relevance").setWidth("120px");
        
        // Context Precision 分数列
        grid.addColumn(new ComponentRenderer<>(result -> {
            double score = result.getContextPrecisionScore();
            Span span = new Span(String.format("%.3f", score));
            if (score >= 0.8) {
                span.addClassName(LumoUtility.TextColor.SUCCESS);
            } else if (score >= 0.6) {
                span.addClassName(LumoUtility.TextColor.WARNING);
            } else {
                span.addClassName(LumoUtility.TextColor.ERROR);
            }
            return span;
        })).setHeader("Context Precision").setWidth("150px");
        
        // Answer Similarity 分数列
        grid.addColumn(new ComponentRenderer<>(result -> {
            double score = result.getAnswerSimilarityScore();
            Span span = new Span(String.format("%.3f", score));
            if (score >= 0.8) {
                span.addClassName(LumoUtility.TextColor.SUCCESS);
            } else if (score >= 0.6) {
                span.addClassName(LumoUtility.TextColor.WARNING);
            } else {
                span.addClassName(LumoUtility.TextColor.ERROR);
            }
            return span;
        })).setHeader("Answer Similarity").setWidth("150px");
        
        return grid;
    }
    
    /**
     * 开始评估
     */
    private void startEvaluation() {
        int numQuestions = numQuestionsField.getValue() != null ? numQuestionsField.getValue() : 10;
        
        // 禁用按钮
        startEvaluationButton.setEnabled(false);
        progressBar.setVisible(true);
        progressBar.setIndeterminate(true);
        statusLabel.setVisible(true);
        statusLabel.setText("正在生成测试集...");
        resultsGrid.setVisible(false);
        reportDiv.setVisible(false);
        
        // 在后台线程执行评估
        Mono.fromCallable(() -> {
            return evaluationService.runFullEvaluation(numQuestions);
        })
        .subscribeOn(Schedulers.boundedElastic())
        .doOnSubscribe(subscription -> {
            getUI().ifPresent(ui -> ui.access(() -> {
                statusLabel.setText("正在生成测试集...");
            }));
        })
        .doOnNext(results -> {
            // 更新进度：测试集生成完成
            getUI().ifPresent(ui -> ui.access(() -> {
                statusLabel.setText("正在批量测试...");
                progressBar.setIndeterminate(false);
                progressBar.setValue(0.0);
            }));
        })
        .subscribe(
            results -> {
                // 评估完成
                getUI().ifPresent(ui -> ui.access(() -> {
                    progressBar.setVisible(false);
                    statusLabel.setText(String.format("评估完成！共测试 %d 个问题", results.size()));
                    statusLabel.addClassName(LumoUtility.TextColor.SUCCESS);
                    
                    // 显示结果表格
                    resultsGrid.setItems(results);
                    resultsGrid.setVisible(true);
                    
                    // 显示报告
                    displayReport();
                    
                    // 重新启用按钮
                    startEvaluationButton.setEnabled(true);
                }));
            },
            error -> {
                // 处理错误
                getUI().ifPresent(ui -> ui.access(() -> {
                    progressBar.setVisible(false);
                    statusLabel.setText("评估失败: " + error.getMessage());
                    statusLabel.addClassName(LumoUtility.TextColor.ERROR);
                    startEvaluationButton.setEnabled(true);
                }));
            }
        );
    }
    
    /**
     * 显示评估报告
     */
    private void displayReport() {
        try {
            java.nio.file.Path reportPath = java.nio.file.Paths.get("external-docs/evaluation_report.md");
            if (!java.nio.file.Files.exists(reportPath)) {
                reportDiv.setVisible(false);
                return;
            }
            
            String reportContent = java.nio.file.Files.readString(reportPath);
            
            // 将 Markdown 转换为 HTML
            Node document = markdownParser.parse(reportContent);
            String html = htmlRenderer.render(document);
            
            // 设置 HTML 内容
            reportDiv.getElement().setProperty("innerHTML", html);
            reportDiv.setVisible(true);
            
        } catch (Exception e) {
            // 如果读取失败，隐藏报告区域
            reportDiv.setVisible(false);
        }
    }
}
