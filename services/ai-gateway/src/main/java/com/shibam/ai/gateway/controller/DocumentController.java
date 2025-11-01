package com.shibam.ai.gateway.controller;

import com.shibam.ai.gateway.dto.DocumentAnalysisRequest;
import com.shibam.ai.gateway.dto.DocumentAnalysisResponse;
import com.shibam.ai.gateway.dto.ChatRequest;
import com.shibam.ai.gateway.dto.ChatResponse;
import com.shibam.ai.gateway.service.DocumentService;
import com.shibam.ai.gateway.service.RagService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import jakarta.validation.Valid;
import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
@Slf4j
public class DocumentController {

    private final DocumentService documentService;
    private final RagService ragService;

    @PostMapping(value = "/documents/analyze", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public CompletableFuture<ResponseEntity<DocumentAnalysisResponse>> analyzeDocument(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "analysis_type", defaultValue = "full_intelligence") String analysisType) {
        
        log.info("Received document analysis request: filename={}, size={}, type={}", 
                file.getOriginalFilename(), file.getSize(), analysisType);

        DocumentAnalysisRequest request = DocumentAnalysisRequest.builder()
                .file(file)
                .analysisType(analysisType)
                .build();

        return documentService.analyzeDocument(request)
                .thenApply(response -> {
                    log.info("Document analysis completed: documentId={}, confidence={}", 
                            response.getDocumentId(), response.getConfidence());
                    return ResponseEntity.ok(response);
                })
                .exceptionally(throwable -> {
                    log.error("Document analysis failed", throwable);
                    return ResponseEntity.internalServerError().build();
                });
    }

    @PostMapping("/ai/chat")
    public CompletableFuture<ResponseEntity<ChatResponse>> chatWithDocuments(
            @Valid @RequestBody ChatRequest request) {
        
        log.info("Received chat request: question={}, documentIds={}, model={}", 
                request.getQuestion(), request.getDocumentIds(), request.getModel());

        return ragService.processQuery(request)
                .thenApply(response -> {
                    log.info("Chat response generated: confidence={}, sources={}", 
                            response.getConfidence(), response.getSources().size());
                    return ResponseEntity.ok(response);
                })
                .exceptionally(throwable -> {
                    log.error("Chat processing failed", throwable);
                    return ResponseEntity.internalServerError().build();
                });
    }

    @GetMapping("/documents/{documentId}")
    public ResponseEntity<DocumentAnalysisResponse> getDocument(@PathVariable String documentId) {
        log.info("Retrieving document: documentId={}", documentId);
        
        return documentService.getDocumentById(documentId)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("AI Gateway is healthy");
    }
}