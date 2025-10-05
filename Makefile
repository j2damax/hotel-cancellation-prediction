# Makefile - Developer Convenience Commands
# Usage: make <target> [ARGS=...] [PORT=...] [CV=...] [ROWS=...]

PYTHON ?= python
PORT ?= 8000
MLFLOW_PORT ?= 5000
CV ?= 5
ROWS ?=
CATEGORICAL ?= target

# Docker / Image settings
REGISTRY ?=
IMAGE_NAME ?= hotel-cancellation-prediction
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
IMAGE_FULL := $(if $(REGISTRY),$(REGISTRY)/$(IMAGE_NAME),$(IMAGE_NAME))

# Composite tags to push (sha + latest)
PUSH_TAGS ?= $(IMAGE_TAG) latest

# If ROWS provided, add --limit-rows flag
LIMIT_ROWS_ARG := $(if $(ROWS),--limit-rows $(ROWS),)

.PHONY: help train fast-train api mlflow test clean-artifacts artifacts-status interpretability docker-build docker-push docker-release docker-run deploy-hf-space

help:
	@echo "Available targets:"
	@echo "  make train           - Full training with CV (CV=$(CV))"
	@echo "  make fast-train      - Quick smoke training (limit rows) e.g. ROWS=500"
	@echo "  make api             - Start FastAPI (port $(PORT))"
	@echo "  make mlflow          - Launch MLflow UI (port $(MLFLOW_PORT))"
	@echo "  make test            - Run pytest suite"
	@echo "  make interpretability- Curl interpretability endpoint (API must be running)"
	@echo "  make artifacts-status- List key artifact files"
	@echo "  make clean-artifacts - Remove generated artifact files (careful)"
	@echo "  make docker-build     - Build Docker image (REGISTRY=<registry>)"
	@echo "  make docker-push      - Push image with tags (REGISTRY=<registry>)"
	@echo "  make docker-release   - Build + push (convenience)"
	@echo "  make docker-run       - Run local container exposing API port"
	@echo "  make deploy-hf-space  - Deploy to Hugging Face Spaces"

train:
	$(PYTHON) scripts/train.py --cv-folds $(CV) --categorical-strategy $(CATEGORICAL) $(LIMIT_ROWS_ARG)

fast-train:
	@if [ -z "$(ROWS)" ]; then echo "ERROR: specify ROWS=<n> (e.g. make fast-train ROWS=500)"; exit 1; fi
	$(PYTHON) scripts/train.py --cv-folds 3 --categorical-strategy $(CATEGORICAL) --limit-rows $(ROWS)

api:
	uvicorn main:app --host 0.0.0.0 --port $(PORT)

mlflow:
	mlflow ui --port $(MLFLOW_PORT)

docker-build:
	docker build -t $(IMAGE_FULL):$(IMAGE_TAG) .
	@if echo "$(PUSH_TAGS)" | grep -qw latest; then docker tag $(IMAGE_FULL):$(IMAGE_TAG) $(IMAGE_FULL):latest; fi

docker-push: docker-build
	@for t in $(PUSH_TAGS); do \
	  echo "Pushing $(IMAGE_FULL):$$t"; \
	  docker push $(IMAGE_FULL):$$t; \
	done

docker-release: docker-push
	@echo "Release complete: $(IMAGE_FULL):$(IMAGE_TAG) + extra tags ($(PUSH_TAGS))"

docker-run:
	docker run --rm -p $(PORT):8000 $(IMAGE_FULL):$(IMAGE_TAG)

interpretability:
	curl -s http://localhost:$(PORT)/model/interpretability | head -n 40

test:
	pytest -q

artifacts-status:
	@echo "Artifacts present:" && ls -1 artifacts | sed 's/^/  - /'

clean-artifacts:
	rm -f artifacts/cv_metrics.json \
		artifacts/champion_meta.json \
		artifacts/confusion_matrix.png \
		artifacts/roc_curve.json \
		artifacts/pr_curve.json \
		artifacts/threshold_sweep.csv \
		artifacts/classification_report.json \
		artifacts/shap_summary.png \
		artifacts/shap_importance_bar.png \
		artifacts/feature_importance.json \
		artifacts/shap_values_sample.json \
		artifacts/feature_name_map.json || true
	@echo "Artifacts cleaned."

deploy-hf-space:
	@echo "Deploying to Hugging Face Spaces (clears cache & force updates)..."
	$(PYTHON) scripts/deploy_to_hf_space.py
