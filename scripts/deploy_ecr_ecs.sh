#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/deploy_ecr_ecs.sh <aws-region> <ecr-repo-name> [image-tag]
# Example: ./scripts/deploy_ecr_ecs.sh us-east-1 hotel-cancellation-prediction 20251005

REGION=${1:?"Region required"}
REPO=${2:?"ECR repository name required"}
TAG=${3:-latest}

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}"

 echo "[1/7] Ensuring ECR login"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "[2/7] Building image" 
docker build -t ${REPO}:${TAG} .

echo "[3/7] Tagging image for ECR"
docker tag ${REPO}:${TAG} ${ECR_URI}

echo "[4/7] Pushing image to ECR"
docker push ${ECR_URI}

echo "[5/7] Rendering task definition JSON (ecs-task-definition.template.json)"
TEMPLATE=ecs-task-definition.template.json
OUTPUT=ecs-task-definition.rendered.json
cp "$TEMPLATE" "$OUTPUT"
sed -i '' -e "s#<ACCOUNT_ID>#${ACCOUNT_ID}#g" "$OUTPUT"
sed -i '' -e "s#<AWS_REGION>#${REGION}#g" "$OUTPUT"

# Optionally inject image with tag if not 'latest'
if [[ "$TAG" != "latest" ]]; then
  sed -i '' -e "s#:latest#:${TAG}#g" "$OUTPUT"
fi

echo "[6/7] Registering task definition"
TASK_DEF_ARN=$(aws ecs register-task-definition --cli-input-json file://$OUTPUT --query 'taskDefinition.taskDefinitionArn' --output text)
echo "Registered task definition: $TASK_DEF_ARN"

echo "[7/7] (Manual) Update or create service referencing new task definition"
echo "Use: aws ecs update-service --cluster <cluster> --service <service> --task-definition $TASK_DEF_ARN --region $REGION"

echo "Done."
