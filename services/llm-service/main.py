from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import asyncio
import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Service",
    description="Large Language Model serving service with multiple model support",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str
    model: str = "llama-2-7b"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class GenerateResponse(BaseModel):
    text: str
    model: str
    tokens_used: int
    finish_reason: str
    timestamp: datetime

class ModelInfo(BaseModel):
    name: str
    type: str
    parameters: str
    loaded: bool
    memory_usage: str

# Model Manager
class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_configs = {
            "llama-2-7b": {
                "model_path": "meta-llama/Llama-2-7b-chat-hf",
                "type": "causal-lm",
                "parameters": "7B",
                "requires_gpu": True
            },
            "gpt2-medium": {
                "model_path": "gpt2-medium", 
                "type": "causal-lm",
                "parameters": "355M",
                "requires_gpu": False
            },
            "distilgpt2": {
                "model_path": "distilgpt2",
                "type": "causal-lm", 
                "parameters": "82M",
                "requires_gpu": False
            }
        }
        self.loaded_models = set()
    
    async def load_model(self, model_name: str):
        """Load a model into memory"""
        if model_name in self.loaded_models:
            return
        
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        config = self.model_configs[model_name]
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            device = "cuda" if torch.cuda.is_available() and config["requires_gpu"] else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                config["model_path"],
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                model = model.to(device)
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.loaded_models.add(model_name)
            
            logger.info(f"Model {model_name} loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    async def generate_text(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text using specified model"""
        model_name = request.model
        
        # Load model if not already loaded
        if model_name not in self.loaded_models:
            await self.load_model(model_name)
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        try:
            # Tokenize input
            inputs = tokenizer.encode(request.prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(request.prompt):
                generated_text = generated_text[len(request.prompt):].strip()
            
            return GenerateResponse(
                text=generated_text,
                model=model_name,
                tokens_used=len(outputs[0]) - len(inputs[0]),
                finish_reason="stop",
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Generation failed for model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_model_info(self) -> List[ModelInfo]:
        """Get information about available models"""
        info_list = []
        for name, config in self.model_configs.items():
            memory_usage = "Unknown"
            if name in self.loaded_models:
                try:
                    model = self.models[name]
                    memory_usage = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "CPU"
                except:
                    memory_usage = "Error"
            
            info_list.append(ModelInfo(
                name=name,
                type=config["type"],
                parameters=config["parameters"],
                loaded=name in self.loaded_models,
                memory_usage=memory_usage
            ))
        
        return info_list

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting LLM Service...")
    
    # Load default model
    try:
        await model_manager.load_model("distilgpt2")
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load default model: {str(e)}")
    
    logger.info("LLM Service started successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "loaded_models": list(model_manager.loaded_models),
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using specified model"""
    try:
        logger.info(f"Generating text with model: {request.model}")
        response = await model_manager.generate_text(request)
        logger.info(f"Generated {response.tokens_used} tokens")
        return response
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models and their status"""
    return model_manager.get_model_info()

@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific model"""
    try:
        await model_manager.load_model(model_name)
        return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """Unload a specific model"""
    try:
        if model_name in model_manager.loaded_models:
            del model_manager.models[model_name]
            del model_manager.tokenizers[model_name]
            model_manager.loaded_models.remove(model_name)
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {"message": f"Model {model_name} unloaded successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    metrics = {
        "loaded_models": len(model_manager.loaded_models),
        "available_models": len(model_manager.model_configs),
        "gpu_available": torch.cuda.is_available(),
        "models_info": model_manager.get_model_info()
    }
    
    if torch.cuda.is_available():
        metrics["gpu_memory"] = {
            "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        }
    
    return metrics

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )