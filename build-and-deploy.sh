#!/bin/bash

# AI Document Intelligence Platform - Build and Deploy Script
# This script builds all services and deploys the complete platform

set -e

echo "🚀 Starting AI Document Intelligence Platform Build and Deploy"
echo "=============================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Java
    if ! command -v java &> /dev/null; then
        print_warning "Java is not installed - required for local development"
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_warning "Python 3 is not installed - required for local development"
    fi
    
    # Check kubectl (optional)
    if command -v kubectl &> /dev/null; then
        print_status "kubectl found - Kubernetes deployment available"
    else
        print_warning "kubectl not found - Kubernetes deployment not available"
    fi
    
    print_success "Prerequisites check completed"
}

# Build Java services
build_java_services() {
    print_status "Building Java services..."
    
    cd services/ai-gateway
    
    # Build with Maven
    if [ -f "./mvnw" ]; then
        print_status "Building AI Gateway with Maven Wrapper..."
        chmod +x ./mvnw
        ./mvnw clean package -DskipTests
    else
        print_status "Building AI Gateway with Maven..."
        mvn clean package -DskipTests
    fi
    
    # Build Docker image
    print_status "Building AI Gateway Docker image..."
    docker build -t shibam/ai-gateway:latest .
    
    cd ../..
    print_success "Java services built successfully"
}

# Build Python services
build_python_services() {
    print_status "Building Python services..."
    
    # Build RAG Service
    cd services/rag-service
    print_status "Building RAG Service Docker image..."
    docker build -t shibam/rag-service:latest .
    cd ../..
    
    # Build LLM Service
    cd services/llm-service
    print_status "Building LLM Service Docker image..."
    docker build -t shibam/llm-service:latest .
    cd ../..
    
    # Build MLOps Service
    cd services/mlops-service
    print_status "Building MLOps Service Docker image..."
    docker build -t shibam/mlops-service:latest .
    cd ../..
    
    # Build Document Processor Service
    cd services/document-processor
    print_status "Building Document Processor Service Docker image..."
    docker build -t shibam/document-processor:latest .
    cd ../..
    
    print_success "Python services built successfully"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    # Java tests
    if [ -f "services/ai-gateway/mvnw" ]; then
        cd services/ai-gateway
        print_status "Running Java tests..."
        ./mvnw test
        cd ../..
    fi
    
    # Python tests
    print_status "Running Python tests..."
    
    # Install test dependencies and run tests for each service
    for service in rag-service llm-service mlops-service document-processor; do
        if [ -f "services/$service/test_main.py" ]; then
            print_status "Running tests for $service..."
            cd services/$service
            
            # Create virtual environment if it doesn't exist
            if [ ! -d "venv" ]; then
                python3 -m venv venv
            fi
            
            # Activate virtual environment and install dependencies
            source venv/bin/activate
            pip install -r requirements.txt
            pip install pytest pytest-asyncio httpx
            
            # Run tests
            python -m pytest test_main.py -v
            
            deactivate
            cd ../..
        fi
    done
    
    print_success "All tests completed"
}

# Deploy with Docker Compose
deploy_local() {
    print_status "Deploying locally with Docker Compose..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please update .env file with your API keys before running services"
    fi
    
    # Start services
    print_status "Starting all services..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    print_status "Checking service health..."
    
    services=(
        "http://localhost:8080/api/v1/health:AI Gateway"
        "http://localhost:8001/health:RAG Service"
        "http://localhost:8002/health:LLM Service"
        "http://localhost:8003/health:Document Processor"
        "http://localhost:8004/health:MLOps Service"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service"
        if curl -f -s "$url" > /dev/null; then
            print_success "$name is healthy"
        else
            print_warning "$name is not responding"
        fi
    done
    
    print_success "Local deployment completed"
    print_status "Services are available at:"
    echo "  - AI Gateway: http://localhost:8080"
    echo "  - RAG Service: http://localhost:8001"
    echo "  - LLM Service: http://localhost:8002"
    echo "  - Document Processor: http://localhost:8003"
    echo "  - MLOps Service: http://localhost:8004"
    echo "  - MLflow UI: http://localhost:5000"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    print_status "Deploying to Kubernetes..."
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is required for Kubernetes deployment"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    print_status "Applying Kubernetes manifests..."
    kubectl apply -f infrastructure/k8s/
    
    # Wait for deployments
    print_status "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/ai-gateway
    kubectl wait --for=condition=available --timeout=300s deployment/rag-service
    
    # Get service URLs
    print_status "Getting service URLs..."
    kubectl get services
    
    print_success "Kubernetes deployment completed"
}

# Provision infrastructure with Terraform
provision_infrastructure() {
    print_status "Provisioning infrastructure with Terraform..."
    
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is required for infrastructure provisioning"
        exit 1
    fi
    
    cd infrastructure/terraform
    
    # Initialize Terraform
    print_status "Initializing Terraform..."
    terraform init
    
    # Plan infrastructure
    print_status "Planning infrastructure..."
    terraform plan
    
    # Apply infrastructure (with confirmation)
    read -p "Do you want to apply the infrastructure changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Applying infrastructure..."
        terraform apply -auto-approve
        print_success "Infrastructure provisioned successfully"
    else
        print_status "Infrastructure provisioning skipped"
    fi
    
    cd ../..
}

# Clean up
cleanup() {
    print_status "Cleaning up..."
    
    # Stop Docker Compose services
    docker-compose down
    
    # Remove Docker images (optional)
    read -p "Do you want to remove Docker images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rmi shibam/ai-gateway:latest || true
        docker rmi shibam/rag-service:latest || true
        docker rmi shibam/llm-service:latest || true
        docker rmi shibam/mlops-service:latest || true
        docker rmi shibam/document-processor:latest || true
        print_success "Docker images removed"
    fi
    
    print_success "Cleanup completed"
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build all services"
    echo "  test        Run all tests"
    echo "  deploy      Deploy locally with Docker Compose"
    echo "  k8s         Deploy to Kubernetes"
    echo "  infra       Provision infrastructure with Terraform"
    echo "  cleanup     Clean up resources"
    echo "  all         Build, test, and deploy locally"
    echo ""
    echo "Examples:"
    echo "  $0 all              # Build, test, and deploy everything"
    echo "  $0 build            # Build all services"
    echo "  $0 deploy           # Deploy locally"
    echo "  $0 k8s              # Deploy to Kubernetes"
}

# Main execution
main() {
    case "${1:-all}" in
        "build")
            check_prerequisites
            build_java_services
            build_python_services
            ;;
        "test")
            run_tests
            ;;
        "deploy")
            deploy_local
            ;;
        "k8s")
            deploy_kubernetes
            ;;
        "infra")
            provision_infrastructure
            ;;
        "cleanup")
            cleanup
            ;;
        "all")
            check_prerequisites
            build_java_services
            build_python_services
            run_tests
            deploy_local
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"