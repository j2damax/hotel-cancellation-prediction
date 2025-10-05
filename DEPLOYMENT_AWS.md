# AWS Deployment Skeleton (Bootstrap)

This file restores a minimal AWS-oriented deployment reference while keeping the main branch focused on Hugging Face Spaces for runtime. Use this as a starting point when re-introducing AWS infra.

## 1. Naming Conventions

| Resource | Suggested Name | Notes |
|----------|----------------|------|
| S3 Bucket (models) | hotel-cancel-models-prod | Global unique; store versioned model artifacts |
| ECR Repository | hotel-cancel-api | Holds API container images |
| ECS Cluster | hotel-cancel-cluster | Fargate or EC2 launch type |
| ECS Service | hotel-cancel-service | Runs desired task count |
| Log Group | /ecs/hotel-cancel-api | CloudWatch logs |

Versioned model objects live under: `s3://hotel-cancel-models-prod/models/<version>/` with a `latest.txt` pointer.

## 2. Required IAM Actions (Task Role / User)

Minimum for model fetch & metrics emission:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::hotel-cancel-models-prod",
        "arn:aws:s3:::hotel-cancel-models-prod/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["cloudwatch:PutMetricData"],
      "Resource": "*"
    }
  ]
}
```

Optionally scope ECR actions to specific repository ARN.

## 3. Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| MODEL_S3_URI | Base S3 URI to version root | s3://hotel-cancel-models-prod/models |
| MODEL_VERSION | Specific version or `latest` sentinel | latest |
| DECISION_THRESHOLD | Override champion threshold | 0.35 |
| AWS_REGION | Region for boto3 client | us-east-1 |
| LOG_LEVEL | App logging level | INFO |
| ENABLE_DEBUG_ENDPOINT | Expose /debug/model (avoid prod) | false |

## 4. Publish Model Artifacts (Recap)

Training produces: `models/champion_model.pkl`, `models/preprocessor.pkl`, `artifacts/champion_meta.json`.

Publish:
```bash
python scripts/train.py --cv-folds 5 --categorical-strategy target  # optional fresh run
python scripts/publish_model.py --bucket hotel-cancel-models-prod --prefix models
```
Result:
```
Publish complete:
  bucket:       hotel-cancel-models-prod
  base_prefix:  models
  version:      2025-10-05T12_26_19_676359_00_00
  latest.txt -> 2025-10-05T12_26_19_676359_00_00
```

## 5. Runtime Fetch (main.py Behavior)

At startup if `MODEL_S3_URI` is set:
1. Resolve version (download `latest.txt` if MODEL_VERSION=latest).
2. Download `champion_model.pkl`, `preprocessor.pkl`, `champion_meta.json` to `models/remote/<version>/` (cached).
3. Set `model_version` globally; derive threshold from env override > champion_meta > default 0.5.
4. `/health` returns `model_version` & active threshold.

Reload without redeploy:
```bash
curl -X POST $HOST/model/reload -H 'Content-Type: application/json' -d '{"version":"latest"}'
```

## 6. Docker Image (Lean Runtime)

Keep image free of baked model artifacts; rely on S3 at launch. Ensure network access to S3 and correct IAM role or secrets injection. The existing Dockerfile already excludes model files.

## 7. Placeholder Commands (No Secrets)

```bash
# Create ECR repository
aws ecr create-repository --repository-name hotel-cancel-api --region us-east-1

# Build & push image
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
REPO=hotel-cancel-api

docker build -t ${REPO}:latest .
docker tag ${REPO}:latest ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest

# (ECS) Register task definition (pseudo)
# aws ecs register-task-definition --cli-input-json file://task-def.json
```

## 8. Health & Observability

Endpoints:
- `/health` – status + model_version
- `/metrics` – uptime, counts, latency
- `/model/reload` – dynamic reload / threshold override
- `/model/interpretability` – SHAP metadata (if artifacts present)

## 9. Rollback Strategy

1. Set `latest.txt` to prior version:
```bash
echo "2025-10-05T11_16_52_449001_00_00" | aws s3 cp - s3://hotel-cancel-models-prod/models/latest.txt
```
2. Call reload endpoint or restart tasks.

## 10. Security Notes
- Prefer IAM roles over static keys inside containers.
- Do not enable `/debug/model` in production (set ENABLE_DEBUG_ENDPOINT=false).
- Consider adding integrity hash manifest for model artifacts (future enhancement).

---
This skeleton intentionally stays minimal; extend with IaC (Terraform/CloudFormation) and CI/CD pipeline when ready.
