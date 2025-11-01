# 🚀 AI Document Intelligence Platform - Deployment Guide

## 📋 Prerequisites

### Required Software
- **Docker Desktop** (with Kubernetes enabled)
- **Java 17+** (OpenJDK recommended)
- **Python 3.11+**
- **Node.js 18+** (for frontend development)
- **Git**

### Optional Tools
- **kubectl** (for Kubernetes deployment)
- **Terraform** (for infrastructure provisioning)
- **Maven 3.8+** (if not using wrapper)

## 🏗️ Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/shibam-max/ai-document-intelligence-platform.git
cd ai-document-intelligence-platform

# Make build script executable
chmod +x build-and-deploy.sh

# Copy environment template
cp .env.example .env
```

### 2. Configure Environment
Edit `.env` file with your API keys:
```bash
# AI/ML API Keys
OPENAI_API_KEY=your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
HUGGINGFACE_TOKEN=your-huggingface-token-here

# Database Configuration
POSTGRES_PASSWORD=your-secure-password
```

### 3. Build and Deploy Everything
```bash
# Build all services, run tests, and deploy locally
./build-and-deploy.sh all
```

## 🐳 Docker Deployment

### Local Development
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Service Management
```bash
# Build specific service
docker build -t shibam/ai-gateway:latest services/ai-gateway/

# Run specific service
docker run -p 8080:8080 shibam/ai-gateway:latest
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (local or cloud)
- kubectl configured
- Docker images built and pushed to registry

### Deploy to Kubernetes
```bash
# Deploy all services
kubectl apply -f infrastructure/k8s/

# Check deployment status
kubectl get pods
kubectl get services

# Scale services
kubectl scale deployment ai-gateway --replicas=5
```

### Monitor Deployments
```bash
# Watch pods
kubectl get pods -w

# Check logs
kubectl logs -f deployment/ai-gateway

# Port forward for local access
kubectl port-forward service/ai-gateway-service 8080:8080
```

## ☁️ Cloud Infrastructure

### AWS Deployment with Terraform
```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Plan infrastructure
terraform plan

# Apply infrastructure
terraform apply
```

### Infrastructure Components
- **EKS Cluster** with GPU nodes
- **RDS PostgreSQL** with encryption
- **ElastiCache Redis** cluster
- **S3 Bucket** for document storage
- **VPC** with public/private subnets
- **Security Groups** and IAM roles

## 🔧 Service Configuration

### AI Gateway (Port 8080)
- **Health Check**: `GET /api/v1/health`
- **Document Analysis**: `POST /api/v1/documents/analyze`
- **RAG Chat**: `POST /api/v1/ai/chat`

### RAG Service (Port 8001)
- **Health Check**: `GET /health`
- **Process Document**: `POST /process-document`
- **Chat**: `POST /chat`

### LLM Service (Port 8002)
- **Health Check**: `GET /health`
- **Generate Text**: `POST /generate`
- **List Models**: `GET /models`

### Document Processor (Port 8003)
- **Health Check**: `GET /health`
- **Process Document**: `POST /process`
- **Supported Formats**: `GET /formats`

### MLOps Service (Port 8004)
- **Health Check**: `GET /health`
- **Register Model**: `POST /models/register`
- **Deploy Model**: `POST /models/deploy`

## 📊 Monitoring and Observability

### Access Dashboards
- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Health Checks
```bash
# Check all services
curl http://localhost:8080/api/v1/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
```

## 🧪 Testing

### Run All Tests
```bash
./build-and-deploy.sh test
```

### Java Tests
```bash
cd services/ai-gateway
./mvnw test
```

### Python Tests
```bash
cd services/rag-service
python -m pytest test_main.py -v
```

## 🔍 Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check Docker logs
docker-compose logs service-name

# Check resource usage
docker stats

# Restart specific service
docker-compose restart service-name
```

#### Database Connection Issues
```bash
# Check PostgreSQL
docker-compose exec postgres psql -U ai_user -d ai_platform

# Check Redis
docker-compose exec redis redis-cli ping
```

#### Memory Issues
```bash
# Increase Docker memory limit
# Docker Desktop -> Settings -> Resources -> Memory

# Check Java heap size
export JAVA_OPTS="-Xmx2g -Xms1g"
```

### Performance Tuning

#### Java Services
```yaml
# In docker-compose.yml
environment:
  - JAVA_OPTS=-Xmx2g -Xms1g -XX:+UseG1GC
```

#### Python Services
```yaml
# In docker-compose.yml
environment:
  - WORKERS=4
  - MAX_REQUESTS=1000
```

## 🔐 Security Configuration

### Production Security Checklist
- [ ] Change default passwords
- [ ] Enable HTTPS/TLS
- [ ] Configure JWT secrets
- [ ] Set up API rate limiting
- [ ] Enable audit logging
- [ ] Configure network policies
- [ ] Set up secrets management

### Environment Variables
```bash
# Security
JWT_SECRET=your-jwt-secret-key-here
JWT_EXPIRATION=86400
CORS_ALLOWED_ORIGINS=https://yourdomain.com

# Database
POSTGRES_PASSWORD=secure-password-here
REDIS_PASSWORD=secure-redis-password
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale with Docker Compose
docker-compose up -d --scale ai-gateway=3 --scale rag-service=2

# Scale with Kubernetes
kubectl scale deployment ai-gateway --replicas=5
```

### Auto-scaling (Kubernetes)
```yaml
# HPA is already configured in k8s manifests
# Scales based on CPU and memory usage
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-gateway-hpa
spec:
  minReplicas: 3
  maxReplicas: 10
```

## 🚀 Production Deployment

### Pre-deployment Checklist
- [ ] All tests passing
- [ ] Security review completed
- [ ] Performance testing done
- [ ] Monitoring configured
- [ ] Backup strategy in place
- [ ] Rollback plan prepared

### Deployment Steps
1. **Build and test** all services
2. **Push images** to container registry
3. **Apply infrastructure** changes
4. **Deploy services** with zero downtime
5. **Verify deployment** with health checks
6. **Monitor** system performance

## 📞 Support

### Getting Help
- **Documentation**: Check README.md files in each service
- **Logs**: Use `docker-compose logs` or `kubectl logs`
- **Health Checks**: Monitor service endpoints
- **Metrics**: Use Grafana dashboards

### Contact Information
- **Developer**: Shibam Samaddar
- **Email**: shibamsamaddar1999@gmail.com
- **GitHub**: https://github.com/shibam-max

---

**🎉 Your AI Document Intelligence Platform is ready for production!**