package com.example.ai.controller;

import com.example.ai.service.ChatService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import java.util.Map;

/**
 * REST API 控制器 - 提供 HTTP 接口
 * 该控制器仅负责 HTTP 请求处理，所有 AI 逻辑委托给 ChatService
 */
@RestController
@RequestMapping("/api/ai")
public class AiChatController {

    private final ChatService chatService;

    public AiChatController(ChatService chatService) {
        this.chatService = chatService;
    }

    /**
     * 同步聊天接口
     */
    @GetMapping("/chat")
    public String chat(@RequestParam String prompt) {
        try {
            return chatService.getResponse(prompt);
        } catch (IllegalArgumentException e) {
            return "错误：" + e.getMessage();
        } catch (Exception e) {
            return "错误：处理请求时发生异常 - " + e.getMessage();
        }
    }

    /**
     * 流式聊天接口 - 返回 SSE 格式的响应流
     */
    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> streamChat(@RequestBody Map<String, String> request) {
        String prompt = request.get("prompt");
        
        return chatService.streamResponse(prompt)
                .map(content -> "data: " + content + "\n\n")
                .onErrorResume(error -> {
                    String errorMessage = "错误：" + error.getMessage();
                    return Flux.just("data: " + errorMessage + "\n\n");
                });
    }
}

