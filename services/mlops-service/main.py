from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from datetime import datetime
import asyncio
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Service",
    description="ML model lifecycle management and deployment service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ModelRegistrationRequest(BaseModel):
    name: str
    version: str
    model_type: str
    description: Optional[str] = None
    tags: Dict[str, str] = {}
    metrics: Dict[str, float] = {}

class ModelDeploymentRequest(BaseModel):
    model_name: str
    version: str
    stage: str = "Production"
    description: Optional[str] = None

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    timestamp: datetime

class ExperimentRequest(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Dict[str, str] = {}

class ModelInfo(BaseModel):
    name: str
    version: str
    stage: str
    description: str
    metrics: Dict[str, float]
    tags: Dict[str, str]
    created_at: datetime
    updated_at: datetime

# MLOps Manager
class MLOpsManager:
    def __init__(self):
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = mlflow.tracking.MlflowClient()
        self.deployed_models = {}
        
    async def initialize(self):
        """Initialize MLOps service"""
        try:
            # Test MLflow connection
            experiments = self.client.search_experiments()
            logger.info(f"Connected to MLflow at {self.mlflow_uri}")
            logger.info(f"Found {len(experiments)} experiments")
        except Exception as e:
            logger.warning(f"MLflow connection failed: {str(e)}")
    
    async def create_experiment(self, request: ExperimentRequest) -> str:
        """Create a new MLflow experiment"""
        try:
            experiment_id = mlflow.create_experiment(
                name=request.name,
                tags=request.tags
            )
            logger.info(f"Created experiment: {request.name} with ID: {experiment_id}")
            return experiment_id
        except Exception as e:
            logger.error(f"Failed to create experiment: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def register_model(self, request: ModelRegistrationRequest) -> str:
        """Register a model in MLflow Model Registry"""
        try:
            # Create model version
            model_version = self.client.create_model_version(
                name=request.name,
                source=f"models:/{request.name}/{request.version}",
                description=request.description,
                tags=request.tags
            )
            
            # Log metrics if provided
            if request.metrics:
                run_id = model_version.run_id
                if run_id:
                    for metric_name, metric_value in request.metrics.items():
                        self.client.log_metric(run_id, metric_name, metric_value)
            
            logger.info(f"Registered model: {request.name} version {request.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def deploy_model(self, request: ModelDeploymentRequest) -> Dict[str, Any]:
        """Deploy a model to specified stage"""
        try:
            # Transition model to specified stage
            self.client.transition_model_version_stage(
                name=request.model_name,
                version=request.version,
                stage=request.stage,
                description=request.description
            )
            
            # Load model for serving
            model_uri = f"models:/{request.model_name}/{request.stage}"
            
            try:
                # Try to load as PyTorch model first
                model = mlflow.pytorch.load_model(model_uri)
                model_type = "pytorch"
            except:
                try:
                    # Try to load as sklearn model
                    model = mlflow.sklearn.load_model(model_uri)
                    model_type = "sklearn"
                except:
                    model = None
                    model_type = "unknown"
            
            deployment_info = {
                "model_name": request.model_name,
                "version": request.version,
                "stage": request.stage,
                "model_uri": model_uri,
                "model_type": model_type,
                "deployed_at": datetime.utcnow(),
                "status": "deployed" if model else "failed"
            }
            
            if model:
                self.deployed_models[f"{request.model_name}:{request.version}"] = {
                    "model": model,
                    "info": deployment_info
                }
            
            logger.info(f"Deployed model: {request.model_name} v{request.version} to {request.stage}")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_model_info(self, model_name: str) -> List[ModelInfo]:
        """Get information about model versions"""
        try:
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            info_list = []
            for version in model_versions:
                # Get metrics for this version
                metrics = {}
                if version.run_id:
                    run = self.client.get_run(version.run_id)
                    metrics = run.data.metrics
                
                info_list.append(ModelInfo(
                    name=version.name,
                    version=version.version,
                    stage=version.current_stage,
                    description=version.description or "",
                    metrics=metrics,
                    tags=version.tags or {},
                    created_at=datetime.fromtimestamp(version.creation_timestamp / 1000),
                    updated_at=datetime.fromtimestamp(version.last_updated_timestamp / 1000)
                ))
            
            return info_list
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def predict(self, model_name: str, version: str, data: List[List[float]]) -> List[float]:
        """Make predictions using deployed model"""
        model_key = f"{model_name}:{version}"
        
        if model_key not in self.deployed_models:
            raise HTTPException(status_code=404, detail=f"Model {model_key} not deployed")
        
        try:
            model = self.deployed_models[model_key]["model"]
            predictions = model.predict(np.array(data))
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def calculate_metrics(self, y_true: List, y_pred: List) -> ModelMetrics:
        """Calculate model performance metrics"""
        try:
            metrics = ModelMetrics(
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, average='weighted'),
                recall=recall_score(y_true, y_pred, average='weighted'),
                f1_score=f1_score(y_true, y_pred, average='weighted'),
                timestamp=datetime.utcnow()
            )
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments"""
        try:
            experiments = self.client.search_experiments()
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "tags": exp.tags
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Failed to get experiments: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize MLOps manager
mlops_manager = MLOpsManager()

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting MLOps Service...")
    await mlops_manager.initialize()
    logger.info("MLOps Service started successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "mlflow_uri": mlops_manager.mlflow_uri,
        "deployed_models": len(mlops_manager.deployed_models)
    }

@app.post("/experiments")
async def create_experiment(request: ExperimentRequest):
    """Create a new experiment"""
    experiment_id = await mlops_manager.create_experiment(request)
    return {"experiment_id": experiment_id, "name": request.name}

@app.get("/experiments")
async def list_experiments():
    """List all experiments"""
    return await mlops_manager.get_experiments()

@app.post("/models/register")
async def register_model(request: ModelRegistrationRequest):
    """Register a new model version"""
    version = await mlops_manager.register_model(request)
    return {"model_name": request.name, "version": version, "status": "registered"}

@app.post("/models/deploy")
async def deploy_model(request: ModelDeploymentRequest):
    """Deploy a model to specified stage"""
    deployment_info = await mlops_manager.deploy_model(request)
    return deployment_info

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get model information and versions"""
    return await mlops_manager.get_model_info(model_name)

@app.post("/models/{model_name}/{version}/predict")
async def predict(model_name: str, version: str, data: Dict[str, List[List[float]]]):
    """Make predictions using deployed model"""
    predictions = await mlops_manager.predict(model_name, version, data["input"])
    return {"predictions": predictions, "model": f"{model_name}:{version}"}

@app.post("/metrics/calculate")
async def calculate_metrics(data: Dict[str, List]):
    """Calculate model performance metrics"""
    metrics = await mlops_manager.calculate_metrics(data["y_true"], data["y_pred"])
    return metrics

@app.get("/models/deployed")
async def list_deployed_models():
    """List all deployed models"""
    deployed = []
    for key, info in mlops_manager.deployed_models.items():
        deployed.append({
            "key": key,
            "info": info["info"]
        })
    return deployed

@app.get("/metrics")
async def get_service_metrics():
    """Get service metrics"""
    return {
        "deployed_models": len(mlops_manager.deployed_models),
        "mlflow_uri": mlops_manager.mlflow_uri,
        "experiments_count": len(await mlops_manager.get_experiments()),
        "uptime": "running"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )