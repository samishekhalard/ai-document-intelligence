# ─────────────────────────────────────────────────────────────────────
#  Enterprise AI Document Intelligence Platform — developer Makefile
# ─────────────────────────────────────────────────────────────────────

PYTHON      ?= python3.11
VENV        ?= .venv
PIP          = $(VENV)/bin/pip
PY           = $(VENV)/bin/python
UVICORN      = $(VENV)/bin/uvicorn
STREAMLIT    = $(VENV)/bin/streamlit
PYTEST       = $(VENV)/bin/pytest
RUFF         = $(VENV)/bin/ruff
MYPY         = $(VENV)/bin/mypy

API_HOST    ?= 127.0.0.1
API_PORT    ?= 8000
UI_PORT     ?= 8501

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Environment setup ───────────────────────────────────────────────
.PHONY: install
install: $(VENV)/bin/python ## Create venv and install all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	-$(PIP) install -r requirements-dev.txt
	$(PY) -m spacy download xx_ent_wiki_sm || true
	$(PY) -m spacy download en_core_web_sm || true
	@echo "✓ install complete — run 'make run'"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

# ── Run the application ─────────────────────────────────────────────
.PHONY: run
run: ## Start FastAPI (8000) and Streamlit (8501) locally
	@echo "API  → http://$(API_HOST):$(API_PORT)"
	@echo "UI   → http://$(API_HOST):$(UI_PORT)"
	$(UVICORN) backend.api.main:app --host $(API_HOST) --port $(API_PORT) --reload & \
	  $(STREAMLIT) run frontend/app.py --server.port $(UI_PORT) --server.address $(API_HOST); \
	  kill %1 2>/dev/null || true

.PHONY: api
api: ## Start only the FastAPI backend
	$(UVICORN) backend.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

.PHONY: ui
ui: ## Start only the Streamlit frontend
	$(STREAMLIT) run frontend/app.py --server.port $(UI_PORT) --server.address $(API_HOST)

.PHONY: load-samples
load-samples: ## Ingest the 3 bundled sample documents
	$(PY) data/sample_docs/load_samples.py

# ── Tests & evaluation ──────────────────────────────────────────────
.PHONY: test
test: ## Run the pytest suite
	$(PYTEST) -q

.PHONY: test-cov
test-cov: ## Run pytest with coverage (requires pytest-cov)
	$(PYTEST) --cov=backend --cov-report=term-missing

.PHONY: eval
eval: ## Run the 20-case RAG evaluation harness
	$(PY) eval/evaluate.py

# ── Quality gates ───────────────────────────────────────────────────
.PHONY: lint
lint: ## Run ruff + mypy
	-$(RUFF) check backend frontend examples tests eval
	-$(MYPY) backend

.PHONY: format
format: ## Auto-format with ruff
	-$(RUFF) format backend frontend examples tests eval
	-$(RUFF) check --fix backend frontend examples tests eval

# ── Docker ──────────────────────────────────────────────────────────
.PHONY: docker-up
docker-up: ## Build and start the Docker Compose stack
	docker compose up --build

.PHONY: docker-down
docker-down: ## Stop and remove the Docker Compose stack
	docker compose down

.PHONY: docker-logs
docker-logs: ## Tail Docker Compose logs
	docker compose logs -f --tail=100

# ── Housekeeping ────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove caches, logs, chroma, uploads, sqlite
	rm -rf data/chroma data/uploads data/metadata.db
	rm -rf logs/*.log
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ cleaned"

.PHONY: distclean
distclean: clean ## Also remove the virtualenv
	rm -rf $(VENV)
	@echo "✓ removed $(VENV)"
