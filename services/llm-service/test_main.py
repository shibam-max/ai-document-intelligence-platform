import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import torch
from datetime import datetime

from main import app, ModelManager

client = TestClient(app)

class TestLLMService:
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "loaded_models" in data
        assert "gpu_available" in data

    @patch('main.model_manager')
    def test_generate_text_success(self, mock_manager):
        """Test text generation endpoint"""
        mock_response = {
            "text": "This is a generated response about AI and machine learning.",
            "model": "gpt2-medium",
            "tokens_used": 15,
            "finish_reason": "stop",
            "timestamp": datetime.utcnow()
        }
        
        mock_manager.generate_text = AsyncMock(return_value=type('obj', (object,), mock_response))
        
        request_data = {
            "prompt": "Tell me about AI",
            "model": "gpt2-medium",
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = client.post("/generate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["model"] == "gpt2-medium"
        assert "tokens_used" in data

    @patch('main.model_manager')
    def test_list_models(self, mock_manager):
        """Test list models endpoint"""
        mock_models = [
            {
                "name": "gpt2-medium",
                "type": "causal-lm",
                "parameters": "355M",
                "loaded": True,
                "memory_usage": "1.2 GB"
            },
            {
                "name": "llama-2-7b",
                "type": "causal-lm", 
                "parameters": "7B",
                "loaded": False,
                "memory_usage": "Unknown"
            }
        ]
        
        mock_manager.get_model_info.return_value = [type('obj', (object,), model) for model in mock_models]
        
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "gpt2-medium"
        assert data[0]["loaded"] == True

    @patch('main.model_manager')
    def test_load_model_success(self, mock_manager):
        """Test model loading endpoint"""
        mock_manager.load_model = AsyncMock()
        
        response = client.post("/models/distilgpt2/load")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "distilgpt2" in data["message"]

    @patch('main.model_manager')
    def test_unload_model_success(self, mock_manager):
        """Test model unloading endpoint"""
        mock_manager.loaded_models = {"distilgpt2"}
        mock_manager.models = {"distilgpt2": Mock()}
        mock_manager.tokenizers = {"distilgpt2": Mock()}
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                response = client.delete("/models/distilgpt2/unload")
                
                assert response.status_code == 200
                data = response.json()
                assert "unloaded successfully" in data["message"]

    @patch('main.model_manager')
    def test_get_metrics(self, mock_manager):
        """Test metrics endpoint"""
        mock_info = [
            type('obj', (object,), {
                "name": "gpt2-medium",
                "loaded": True,
                "memory_usage": "1.2 GB"
            })
        ]
        mock_manager.get_model_info.return_value = mock_info
        mock_manager.loaded_models = {"gpt2-medium"}
        mock_manager.model_configs = {"gpt2-medium": {}, "llama-2-7b": {}}
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=1024**3):
                with patch('torch.cuda.memory_reserved', return_value=2*1024**3):
                    response = client.get("/metrics")
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["loaded_models"] == 1
                    assert data["available_models"] == 2
                    assert "gpu_memory" in data

class TestModelManager:
    
    @pytest.fixture
    def model_manager(self):
        return ModelManager()

    @pytest.mark.asyncio
    async def test_load_model_success(self, model_manager):
        """Test successful model loading"""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('torch.cuda.is_available', return_value=False):
                    
                    mock_tokenizer.return_value = Mock()
                    mock_tokenizer.return_value.pad_token = None
                    mock_tokenizer.return_value.eos_token = "<eos>"
                    
                    mock_model_instance = Mock()
                    mock_model_instance.to.return_value = mock_model_instance
                    mock_model.return_value = mock_model_instance
                    
                    await model_manager.load_model("distilgpt2")
                    
                    assert "distilgpt2" in model_manager.loaded_models
                    assert "distilgpt2" in model_manager.models
                    assert "distilgpt2" in model_manager.tokenizers

    @pytest.mark.asyncio
    async def test_load_unsupported_model(self, model_manager):
        """Test loading unsupported model"""
        with pytest.raises(ValueError, match="Model unsupported_model not supported"):
            await model_manager.load_model("unsupported_model")

    @pytest.mark.asyncio
    async def test_generate_text_success(self, model_manager):
        """Test text generation"""
        # Setup mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Generated text response"
        mock_tokenizer.eos_token_id = 2
        
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        model_manager.models["test_model"] = mock_model
        model_manager.tokenizers["test_model"] = mock_tokenizer
        model_manager.loaded_models.add("test_model")
        
        request = type('obj', (object,), {
            'model': 'test_model',
            'prompt': 'Test prompt',
            'max_tokens': 50,
            'temperature': 0.7,
            'top_p': 0.9
        })
        
        with patch('torch.no_grad'):
            response = await model_manager.generate_text(request)
            
            assert response.text == "Generated text response"
            assert response.model == "test_model"
            assert response.tokens_used > 0

    def test_get_model_info(self, model_manager):
        """Test getting model information"""
        model_manager.loaded_models.add("distilgpt2")
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=1024**3):
                info_list = model_manager.get_model_info()
                
                assert len(info_list) == len(model_manager.model_configs)
                
                # Find distilgpt2 info
                distilgpt2_info = next(info for info in info_list if info.name == "distilgpt2")
                assert distilgpt2_info.loaded == True
                assert distilgpt2_info.parameters == "82M"

    @pytest.mark.asyncio
    async def test_generate_text_model_not_loaded(self, model_manager):
        """Test text generation with unloaded model"""
        request = type('obj', (object,), {
            'model': 'distilgpt2',
            'prompt': 'Test prompt',
            'max_tokens': 50,
            'temperature': 0.7,
            'top_p': 0.9
        })
        
        # Mock the load_model method to avoid actual loading
        with patch.object(model_manager, 'load_model') as mock_load:
            mock_load.return_value = None
            
            # Mock the actual generation after loading
            mock_model = Mock()
            mock_tokenizer = Mock()
            
            mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
            mock_tokenizer.decode.return_value = "Generated text"
            mock_tokenizer.eos_token_id = 2
            
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            mock_model.parameters.return_value = [torch.tensor([1.0])]
            
            model_manager.models["distilgpt2"] = mock_model
            model_manager.tokenizers["distilgpt2"] = mock_tokenizer
            model_manager.loaded_models.add("distilgpt2")
            
            with patch('torch.no_grad'):
                response = await model_manager.generate_text(request)
                
                mock_load.assert_called_once_with("distilgpt2")
                assert response.text == "Generated text"

class TestModelIntegration:
    """Integration tests for model operations"""
    
    @pytest.mark.asyncio
    async def test_full_generation_workflow(self):
        """Test complete generation workflow"""
        # This test would require actual model files, so we'll mock it
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                
                # Setup mocks
                tokenizer_instance = Mock()
                tokenizer_instance.pad_token = None
                tokenizer_instance.eos_token = "<eos>"
                tokenizer_instance.encode.return_value = torch.tensor([[1, 2, 3]])
                tokenizer_instance.decode.return_value = "Test prompt Generated response"
                tokenizer_instance.eos_token_id = 2
                mock_tokenizer.return_value = tokenizer_instance
                
                model_instance = Mock()
                model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
                model_instance.parameters.return_value = [torch.tensor([1.0])]
                model_instance.to.return_value = model_instance
                mock_model.return_value = model_instance
                
                # Test the workflow
                manager = ModelManager()
                
                with patch('torch.cuda.is_available', return_value=False):
                    await manager.load_model("distilgpt2")
                    
                    request = type('obj', (object,), {
                        'model': 'distilgpt2',
                        'prompt': 'Test prompt',
                        'max_tokens': 50,
                        'temperature': 0.7,
                        'top_p': 0.9
                    })
                    
                    with patch('torch.no_grad'):
                        response = await manager.generate_text(request)
                        
                        assert response.text == "Generated response"  # Prompt removed
                        assert response.model == "distilgpt2"
                        assert response.finish_reason == "stop"

    def test_model_configuration_validation(self):
        """Test model configuration validation"""
        manager = ModelManager()
        
        # Test all configured models have required fields
        for model_name, config in manager.model_configs.items():
            assert "model_path" in config
            assert "type" in config
            assert "parameters" in config
            assert "requires_gpu" in config
            
            # Validate types
            assert isinstance(config["requires_gpu"], bool)
            assert config["type"] in ["causal-lm"]

if __name__ == "__main__":
    pytest.main([__file__])