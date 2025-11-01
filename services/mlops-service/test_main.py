import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import numpy as np

from main import app, MLOpsManager

client = TestClient(app)

class TestMLOpsService:
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "mlflow_uri" in data
        assert "deployed_models" in data

    @patch('main.mlops_manager')
    def test_create_experiment_success(self, mock_manager):
        """Test experiment creation endpoint"""
        mock_manager.create_experiment = AsyncMock(return_value="exp_123")
        
        request_data = {
            "name": "test_experiment",
            "description": "Test experiment for AI models",
            "tags": {"team": "ai", "project": "document_intelligence"}
        }
        
        response = client.post("/experiments", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "exp_123"
        assert data["name"] == "test_experiment"

    @patch('main.mlops_manager')
    def test_register_model_success(self, mock_manager):
        """Test model registration endpoint"""
        mock_manager.register_model = AsyncMock(return_value="v1")
        
        request_data = {
            "name": "document_classifier",
            "version": "1.0.0",
            "model_type": "classification",
            "description": "Document classification model",
            "tags": {"framework": "pytorch"},
            "metrics": {"accuracy": 0.92, "f1_score": 0.89}
        }
        
        response = client.post("/models/register", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "document_classifier"
        assert data["version"] == "v1"
        assert data["status"] == "registered"

    @patch('main.mlops_manager')
    def test_deploy_model_success(self, mock_manager):
        """Test model deployment endpoint"""
        mock_deployment_info = {
            "model_name": "document_classifier",
            "version": "1.0.0",
            "stage": "Production",
            "model_uri": "models:/document_classifier/Production",
            "model_type": "pytorch",
            "deployed_at": datetime.utcnow(),
            "status": "deployed"
        }
        
        mock_manager.deploy_model = AsyncMock(return_value=mock_deployment_info)
        
        request_data = {
            "model_name": "document_classifier",
            "version": "1.0.0",
            "stage": "Production",
            "description": "Deploy to production"
        }
        
        response = client.post("/models/deploy", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "document_classifier"
        assert data["status"] == "deployed"

    @patch('main.mlops_manager')
    def test_get_model_info_success(self, mock_manager):
        """Test get model info endpoint"""
        mock_model_info = [
            {
                "name": "document_classifier",
                "version": "1.0.0",
                "stage": "Production",
                "description": "Document classification model",
                "metrics": {"accuracy": 0.92},
                "tags": {"framework": "pytorch"},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]
        
        mock_manager.get_model_info = AsyncMock(return_value=[
            type('obj', (object,), info) for info in mock_model_info
        ])
        
        response = client.get("/models/document_classifier")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "document_classifier"

    @patch('main.mlops_manager')
    def test_predict_success(self, mock_manager):
        """Test model prediction endpoint"""
        mock_predictions = [0.8, 0.2, 0.9, 0.1]
        mock_manager.predict = AsyncMock(return_value=mock_predictions)
        
        request_data = {
            "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        }
        
        response = client.post("/models/document_classifier/1.0.0/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["model"] == "document_classifier:1.0.0"
        assert len(data["predictions"]) == 4

    @patch('main.mlops_manager')
    def test_calculate_metrics_success(self, mock_manager):
        """Test metrics calculation endpoint"""
        mock_metrics = {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.91,
            "f1_score": 0.90,
            "timestamp": datetime.utcnow()
        }
        
        mock_manager.calculate_metrics = AsyncMock(return_value=type('obj', (object,), mock_metrics))
        
        request_data = {
            "y_true": [1, 0, 1, 1, 0],
            "y_pred": [1, 0, 1, 0, 0]
        }
        
        response = client.post("/metrics/calculate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["accuracy"] == 0.92
        assert data["f1_score"] == 0.90

    @patch('main.mlops_manager')
    def test_list_experiments(self, mock_manager):
        """Test list experiments endpoint"""
        mock_experiments = [
            {
                "experiment_id": "1",
                "name": "document_classification",
                "lifecycle_stage": "active",
                "tags": {"team": "ai"}
            }
        ]
        
        mock_manager.get_experiments = AsyncMock(return_value=mock_experiments)
        
        response = client.get("/experiments")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "document_classification"

class TestMLOpsManager:
    
    @pytest.fixture
    def mlops_manager(self):
        with patch('mlflow.set_tracking_uri'):
            with patch('mlflow.tracking.MlflowClient'):
                return MLOpsManager()

    @pytest.mark.asyncio
    async def test_initialize_success(self, mlops_manager):
        """Test MLOps manager initialization"""
        mock_experiments = [Mock()]
        mlops_manager.client.search_experiments.return_value = mock_experiments
        
        await mlops_manager.initialize()
        
        mlops_manager.client.search_experiments.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_experiment_success(self, mlops_manager):
        """Test experiment creation"""
        with patch('mlflow.create_experiment', return_value="exp_123") as mock_create:
            request = type('obj', (object,), {
                'name': 'test_experiment',
                'tags': {'team': 'ai'}
            })
            
            experiment_id = await mlops_manager.create_experiment(request)
            
            assert experiment_id == "exp_123"
            mock_create.assert_called_once_with(
                name='test_experiment',
                tags={'team': 'ai'}
            )

    @pytest.mark.asyncio
    async def test_register_model_success(self, mlops_manager):
        """Test model registration"""
        mock_version = Mock()
        mock_version.version = "1"
        mock_version.run_id = "run_123"
        
        mlops_manager.client.create_model_version.return_value = mock_version
        
        request = type('obj', (object,), {
            'name': 'test_model',
            'version': '1.0.0',
            'description': 'Test model',
            'tags': {'framework': 'pytorch'},
            'metrics': {'accuracy': 0.92}
        })
        
        version = await mlops_manager.register_model(request)
        
        assert version == "1"
        mlops_manager.client.create_model_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_model_success(self, mlops_manager):
        """Test model deployment"""
        with patch('mlflow.pytorch.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            request = type('obj', (object,), {
                'model_name': 'test_model',
                'version': '1.0.0',
                'stage': 'Production',
                'description': 'Deploy to prod'
            })
            
            deployment_info = await mlops_manager.deploy_model(request)
            
            assert deployment_info['model_name'] == 'test_model'
            assert deployment_info['status'] == 'deployed'
            assert 'test_model:1.0.0' in mlops_manager.deployed_models

    @pytest.mark.asyncio
    async def test_predict_success(self, mlops_manager):
        """Test model prediction"""
        # Setup deployed model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.2])
        
        mlops_manager.deployed_models['test_model:1.0.0'] = {
            'model': mock_model,
            'info': {'model_type': 'sklearn'}
        }
        
        predictions = await mlops_manager.predict('test_model', '1.0.0', [[1, 2], [3, 4]])
        
        assert predictions == [0.8, 0.2]
        mock_model.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_model_not_deployed(self, mlops_manager):
        """Test prediction with undeployed model"""
        with pytest.raises(Exception):  # Should raise HTTPException in actual code
            await mlops_manager.predict('nonexistent_model', '1.0.0', [[1, 2]])

    @pytest.mark.asyncio
    async def test_calculate_metrics_success(self, mlops_manager):
        """Test metrics calculation"""
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 1, 0, 0]
        
        metrics = await mlops_manager.calculate_metrics(y_true, y_pred)
        
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'f1_score')
        assert 0 <= metrics.accuracy <= 1

    @pytest.mark.asyncio
    async def test_get_model_info_success(self, mlops_manager):
        """Test getting model information"""
        mock_version = Mock()
        mock_version.name = "test_model"
        mock_version.version = "1.0.0"
        mock_version.current_stage = "Production"
        mock_version.description = "Test model"
        mock_version.tags = {"framework": "pytorch"}
        mock_version.creation_timestamp = 1640995200000  # 2022-01-01
        mock_version.last_updated_timestamp = 1640995200000
        mock_version.run_id = "run_123"
        
        mock_run = Mock()
        mock_run.data.metrics = {"accuracy": 0.92}
        
        mlops_manager.client.search_model_versions.return_value = [mock_version]
        mlops_manager.client.get_run.return_value = mock_run
        
        info_list = await mlops_manager.get_model_info("test_model")
        
        assert len(info_list) == 1
        assert info_list[0].name == "test_model"
        assert info_list[0].version == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_experiments_success(self, mlops_manager):
        """Test getting experiments"""
        mock_exp = Mock()
        mock_exp.experiment_id = "1"
        mock_exp.name = "test_experiment"
        mock_exp.lifecycle_stage = "active"
        mock_exp.tags = {"team": "ai"}
        
        mlops_manager.client.search_experiments.return_value = [mock_exp]
        
        experiments = await mlops_manager.get_experiments()
        
        assert len(experiments) == 1
        assert experiments[0]["name"] == "test_experiment"

class TestMLOpsIntegration:
    """Integration tests for MLOps workflows"""
    
    @pytest.mark.asyncio
    async def test_full_model_lifecycle(self):
        """Test complete model lifecycle"""
        with patch('mlflow.set_tracking_uri'):
            with patch('mlflow.tracking.MlflowClient') as mock_client:
                with patch('mlflow.create_experiment', return_value="exp_1"):
                    
                    manager = MLOpsManager()
                    
                    # Create experiment
                    exp_request = type('obj', (object,), {
                        'name': 'integration_test',
                        'tags': {'test': 'true'}
                    })
                    
                    exp_id = await manager.create_experiment(exp_request)
                    assert exp_id == "exp_1"
                    
                    # Register model
                    mock_version = Mock()
                    mock_version.version = "1"
                    mock_version.run_id = None
                    mock_client.return_value.create_model_version.return_value = mock_version
                    
                    reg_request = type('obj', (object,), {
                        'name': 'integration_model',
                        'version': '1.0.0',
                        'model_type': 'classification',
                        'description': 'Integration test model',
                        'tags': {},
                        'metrics': {}
                    })
                    
                    version = await manager.register_model(reg_request)
                    assert version == "1"

    def test_metrics_calculation_accuracy(self):
        """Test metrics calculation accuracy"""
        # Perfect predictions
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 1, 0, 0, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        expected_accuracy = accuracy_score(y_true, y_pred)
        expected_precision = precision_score(y_true, y_pred, average='weighted')
        expected_recall = recall_score(y_true, y_pred, average='weighted')
        expected_f1 = f1_score(y_true, y_pred, average='weighted')
        
        assert expected_accuracy == 1.0
        assert expected_precision == 1.0
        assert expected_recall == 1.0
        assert expected_f1 == 1.0

    def test_model_configuration_validation(self):
        """Test model configuration validation"""
        manager = MLOpsManager()
        
        # Test MLflow URI configuration
        assert manager.mlflow_uri is not None
        assert isinstance(manager.deployed_models, dict)

if __name__ == "__main__":
    pytest.main([__file__])