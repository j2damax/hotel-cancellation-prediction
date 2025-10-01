# AWS ECR Deployment Guide

This guide provides step-by-step instructions for deploying the Hotel Cancellation Prediction API to Amazon Elastic Container Registry (ECR) and running it on AWS services.

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Docker installed locally
- Models trained and available in the `models/` and `mlruns/` directories

## Step 1: Train Models Locally

Before deploying, ensure you have trained models:

```bash
python scripts/train.py
```

This will create:
- `models/scaler.pkl` - Feature scaler
- `mlruns/` - MLflow tracking data with trained models

## Step 2: Configure AWS CLI

```bash
# Configure AWS credentials
aws configure

# Verify configuration
aws sts get-caller-identity
```

## Step 3: Create ECR Repository

```bash
# Set variables
AWS_REGION="us-east-1"  # Change to your preferred region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="hotel-cancellation-prediction"

# Create ECR repository
aws ecr create-repository \
    --repository-name ${REPO_NAME} \
    --region ${AWS_REGION} \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256

# Output will include the repository URI
```

## Step 4: Authenticate Docker to ECR

```bash
# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

## Step 5: Build Docker Image

```bash
# Build the Docker image
docker build -t ${REPO_NAME}:latest .

# Verify the image
docker images ${REPO_NAME}
```

## Step 6: Tag and Push to ECR

```bash
# Tag the image for ECR
docker tag ${REPO_NAME}:latest \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest

# Also tag with version number (optional)
docker tag ${REPO_NAME}:latest \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:v1.0.0

# Push to ECR
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:v1.0.0
```

## Step 7: Verify Image in ECR

```bash
# List images in repository
aws ecr describe-images \
    --repository-name ${REPO_NAME} \
    --region ${AWS_REGION}
```

## Deployment Options

### Option A: Deploy to Amazon ECS (Fargate)

#### 1. Create ECS Cluster

```bash
CLUSTER_NAME="hotel-prediction-cluster"

aws ecs create-cluster \
    --cluster-name ${CLUSTER_NAME} \
    --region ${AWS_REGION}
```

#### 2. Create Task Definition

Create a file `task-definition.json`:

```json
{
  "family": "hotel-cancellation-prediction",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/hotel-cancellation-prediction:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hotel-cancellation-prediction",
          "awslogs-region": "YOUR_REGION",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c \"import requests; requests.get('http://localhost:8000/health')\" || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ],
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole"
}
```

Replace placeholders and register:

```bash
aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region ${AWS_REGION}
```

#### 3. Create CloudWatch Log Group

```bash
aws logs create-log-group \
    --log-group-name /ecs/hotel-cancellation-prediction \
    --region ${AWS_REGION}
```

#### 4. Create ECS Service

```bash
# Requires VPC and subnets to be configured
aws ecs create-service \
    --cluster ${CLUSTER_NAME} \
    --service-name hotel-prediction-service \
    --task-definition hotel-cancellation-prediction \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}" \
    --region ${AWS_REGION}
```

### Option B: Deploy to Amazon EKS

#### 1. Create EKS Cluster (if not exists)

```bash
eksctl create cluster \
    --name hotel-prediction-cluster \
    --region ${AWS_REGION} \
    --nodegroup-name standard-workers \
    --node-type t3.medium \
    --nodes 2
```

#### 2. Create Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hotel-cancellation-prediction
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hotel-prediction
  template:
    metadata:
      labels:
        app: hotel-prediction
    spec:
      containers:
      - name: api
        image: YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/hotel-cancellation-prediction:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: hotel-prediction-service
spec:
  type: LoadBalancer
  selector:
    app: hotel-prediction
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

Deploy:

```bash
kubectl apply -f k8s-deployment.yaml
kubectl get services hotel-prediction-service
```

### Option C: Deploy to AWS App Runner

```bash
# Create App Runner service from ECR
aws apprunner create-service \
    --service-name hotel-cancellation-prediction \
    --source-configuration '{
        "ImageRepository": {
            "ImageIdentifier": "'${AWS_ACCOUNT_ID}'.dkr.ecr.'${AWS_REGION}'.amazonaws.com/'${REPO_NAME}':latest",
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8000"
            }
        },
        "AutoDeploymentsEnabled": true
    }' \
    --instance-configuration '{
        "Cpu": "1 vCPU",
        "Memory": "2 GB"
    }' \
    --region ${AWS_REGION}
```

## Testing Deployment

After deployment, test the API:

```bash
# Get the service URL (example for ECS with ALB)
SERVICE_URL="http://your-alb-url.region.elb.amazonaws.com"

# Health check
curl ${SERVICE_URL}/health

# Make a prediction
curl -X POST "${SERVICE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lead_time": 120,
    "arrival_month": 7,
    "stays_weekend_nights": 2,
    "stays_week_nights": 3,
    "adults": 2,
    "children": 1,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "booking_changes": 1,
    "adr": 95.50,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 2
  }'
```

## Continuous Deployment

### Using GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to ECR

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build, tag, and push image to Amazon ECR
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: hotel-cancellation-prediction
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
```

## Monitoring and Logging

### CloudWatch Logs

```bash
# View logs
aws logs tail /ecs/hotel-cancellation-prediction --follow
```

### CloudWatch Metrics

Set up custom metrics for:
- Request rate
- Response time
- Error rate
- Prediction latency

## Cost Optimization

1. **Use Fargate Spot** for non-production environments
2. **Enable Auto Scaling** based on CPU/memory
3. **Use ECR Lifecycle Policies** to clean up old images
4. **Monitor CloudWatch metrics** to right-size resources

## Security Best Practices

1. Enable ECR image scanning
2. Use IAM roles for service authentication
3. Implement VPC endpoints for private ECR access
4. Enable encryption at rest and in transit
5. Regularly update base images
6. Use AWS Secrets Manager for sensitive configuration

## Cleanup

To avoid ongoing charges:

```bash
# Delete ECS service
aws ecs delete-service --cluster ${CLUSTER_NAME} --service hotel-prediction-service --force

# Delete ECS cluster
aws ecs delete-cluster --cluster ${CLUSTER_NAME}

# Delete ECR images
aws ecr batch-delete-image \
    --repository-name ${REPO_NAME} \
    --image-ids imageTag=latest

# Delete ECR repository
aws ecr delete-repository --repository-name ${REPO_NAME} --force
```

## Troubleshooting

### Common Issues

1. **Authentication Failed**: Re-run ECR login command
2. **Image Pull Failed**: Check IAM permissions
3. **Health Check Failed**: Verify model files are included in image
4. **Out of Memory**: Increase task memory allocation

### Useful Commands

```bash
# Check ECS task logs
aws ecs describe-tasks --cluster ${CLUSTER_NAME} --tasks TASK_ID

# Check ECR repository
aws ecr describe-repositories --repository-names ${REPO_NAME}

# Get ECR image details
aws ecr describe-images --repository-name ${REPO_NAME}
```

## Support

For issues or questions, please refer to:
- AWS ECS Documentation: https://docs.aws.amazon.com/ecs/
- AWS ECR Documentation: https://docs.aws.amazon.com/ecr/
- Project Repository: https://github.com/j2damax/hotel-cancellation-prediction
