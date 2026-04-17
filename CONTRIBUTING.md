# Contributing

Thank you for your interest in improving the Enterprise AI Document
Intelligence Platform. This guide covers how to set up a development
environment, the expected code quality bar, and the review process.

## Code of Conduct

Be kind, be concise, be professional. Assume good faith. No harassment
or discriminatory behaviour will be tolerated.

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/ai-document-intelligence.git
cd ai-document-intelligence
```

### 2. System prerequisites

```bash
# macOS
brew install python@3.11 tesseract tesseract-lang poppler

# Ubuntu 22.04+
sudo apt-get install python3.11 python3.11-venv \
  tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus poppler-utils

# Ollama + model weights
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Python environment

```bash
make install          # creates .venv, installs deps, pulls spaCy models
cp .env.example .env  # adjust as needed
```

If you prefer manual setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional, for linters
python -m spacy download xx_ent_wiki_sm
python -m spacy download en_core_web_sm
```

### 4. Run the stack

```bash
make run              # FastAPI on :8000, Streamlit on :8501
```

Or manually, in two terminals:

```bash
uvicorn backend.api.main:app --reload
streamlit run frontend/app.py
```

## Development Workflow

### Branching

Feature branches from `main`:

```
feat/<short-description>
fix/<short-description>
docs/<short-description>
refactor/<short-description>
```

### Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(retrieval): add cross-encoder reranker
fix(ingestion): handle empty PDFs without extractable text
docs(readme): clarify Docker install path
refactor(agents): collapse duplicated retriever init
test(orchestrator): cover max-iteration guard
```

### Code style

- Target Python **3.11**.
- Type hints on every public function; Pydantic models on API
  boundaries — no bare `dict[str, Any]` in public contracts.
- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
  for non-trivial functions.
- Maximum line length: 100 characters.
- Prefer `pathlib.Path` over string paths.
- Prefer `logger.info(f"...")` over string concatenation.
- No `print()` in library code.

Enforced by `make lint`:

```bash
make lint   # runs ruff + mypy
```

### Tests

```bash
make test   # pytest suite, <1 second, no Ollama required
```

New features must ship with tests. Prefer monkey-patching over mocks,
and keep tests deterministic (avoid sleeping, avoid real LLM calls).

### Evaluation

```bash
make eval   # 20-case RAG evaluation harness
```

Please run the evaluation harness before opening a PR that touches
retrieval, chunking, or the LLM prompt. Paste the summary into the PR
description.

## Pull Request Checklist

Before opening a PR:

- [ ] `make test` passes.
- [ ] `make lint` passes.
- [ ] New features include tests.
- [ ] User-facing changes updated in `CHANGELOG.md` under an
      `[Unreleased]` section.
- [ ] README / docs updated if the public surface changed.
- [ ] PR description explains the *why*, not just the *what*.

Open the PR against `main`. CI will run tests + linting. A maintainer
will review within a few business days.

## Architecture Changes

For non-trivial architectural changes (new agents, retrieval strategy
changes, schema changes) please open a **design discussion** issue
first. Use the `feature_request` template and clearly describe:

- What problem are you solving?
- What alternatives did you consider?
- What is the expected impact on performance, maintainability, and
  deployment complexity?

## Reporting Bugs

Use the `bug_report` issue template and include:

- Exact reproduction steps.
- Expected vs. actual behaviour.
- Environment (OS, Python, Ollama version).
- Relevant log excerpt from `logs/app.log`.

## Security

Do **not** file security vulnerabilities as public issues. Instead,
email <mksulty@gmail.com> with details; you will receive acknowledgement
within 72 hours.

## License

By contributing, you agree that your contributions will be licensed
under the project's MIT License.
