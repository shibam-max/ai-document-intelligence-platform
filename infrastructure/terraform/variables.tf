variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "ai-document-platform"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "db_password" {
  description = "Password for the RDS PostgreSQL instance"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
}

variable "pinecone_api_key" {
  description = "Pinecone API key"
  type        = string
  sensitive   = true
}

variable "node_instance_types" {
  description = "Instance types for EKS nodes"
  type        = list(string)
  default     = ["m5.xlarge", "m5.2xlarge"]
}

variable "gpu_instance_types" {
  description = "Instance types for GPU nodes"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge"]
}

variable "min_nodes" {
  description = "Minimum number of nodes"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "desired_nodes" {
  description = "Desired number of nodes"
  type        = number
  default     = 3
}