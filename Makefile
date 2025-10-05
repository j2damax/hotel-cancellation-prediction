# Makefile - Developer Convenience Commands
# Usage: make <target> [ARGS=...] [PORT=...] [CV=...] [ROWS=...]

PYTHON ?= python
PORT ?= 8000
MLFLOW_PORT ?= 5000
CV ?= 5
ROWS ?=
CATEGORICAL ?= target

# If ROWS provided, add --limit-rows flag
LIMIT_ROWS_ARG := $(if $(ROWS),--limit-rows $(ROWS),)

.PHONY: help train fast-train api mlflow test clean-artifacts artifacts-status interpretability

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

train:
	$(PYTHON) scripts/train.py --cv-folds $(CV) --categorical-strategy $(CATEGORICAL) $(LIMIT_ROWS_ARG)

fast-train:
	@if [ -z "$(ROWS)" ]; then echo "ERROR: specify ROWS=<n> (e.g. make fast-train ROWS=500)"; exit 1; fi
	$(PYTHON) scripts/train.py --cv-folds 3 --categorical-strategy $(CATEGORICAL) --limit-rows $(ROWS)

api:
	uvicorn main:app --host 0.0.0.0 --port $(PORT)

mlflow:
	mlflow ui --port $(MLFLOW_PORT)

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
