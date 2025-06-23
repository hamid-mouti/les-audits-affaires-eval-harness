# 🏛️ Les Audits-Affaires Evaluation Harness - Makefile
# Development and deployment automation

.PHONY: help install install-dev test test-unit test-integration lint format format-check \
        cli-test cli-demo eval-small eval-medium eval-full analyze analyze-all \
        build publish publish-test clean dev-cycle quality prod-check \
        test-providers demo-providers setup-providers

# Default target
help: ## Show this help message
	@echo "🏛️ Les Audits-Affaires Evaluation Harness"
	@echo "════════════════════════════════════════"
	@echo ""
	@echo "📦 Installation:"
	@echo "  install           Install package in current environment"
	@echo "  install-dev       Install with development dependencies"
	@echo "  setup-providers   Show how to set up external provider API keys"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-providers    Test external provider connections"
	@echo "  demo-providers    Demo external provider functionality"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  lint              Run all linters (flake8, mypy, bandit)"
	@echo "  format            Format code with black and isort"
	@echo "  format-check      Check code formatting without changes"
	@echo ""
	@echo "🚀 CLI Testing:"
	@echo "  cli-test          Test CLI commands"
	@echo "  cli-demo          Demo CLI functionality"
	@echo ""
	@echo "📊 Evaluation:"
	@echo "  eval-small        Run small evaluation (10 samples)"
	@echo "  eval-medium       Run medium evaluation (100 samples)"
	@echo "  eval-full         Run full evaluation (all samples)"
	@echo ""
	@echo "📈 Analysis:"
	@echo "  analyze           Analyze latest results"
	@echo "  analyze-all       Generate all analysis outputs"
	@echo ""
	@echo "🏗️  Build & Deploy:"
	@echo "  build             Build distribution packages"
	@echo "  publish           Publish to PyPI"
	@echo "  publish-test      Publish to TestPyPI"
	@echo "  clean             Clean build artifacts"
	@echo ""
	@echo "🔄 Development Workflows:"
	@echo "  dev-cycle         Complete development cycle"
	@echo "  quality           Run all quality checks"
	@echo "  prod-check        Production readiness check"

# Installation
install: ## Install the package
	pip install -e .

install-dev: ## Install with development dependencies
	pip install -e ".[dev]"
	pre-commit install

# Testing
test: ## Run all tests
	pytest tests/ -v --cov=les_audits_affaires_eval --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v

# Code Quality
lint: ## Run linting checks
	flake8 src/les_audits_affaires_eval/ tests/
	mypy src/les_audits_affaires_eval/

format: ## Format code with black and isort
	black src/les_audits_affaires_eval/ tests/ scripts/
	isort src/les_audits_affaires_eval/ tests/ scripts/

format-check: ## Check code formatting without making changes
	black --check src/les_audits_affaires_eval/ tests/ scripts/
	isort --check-only src/les_audits_affaires_eval/ tests/ scripts/

# CLI Testing
cli-test: ## Test CLI functionality
	lae-eval info
	@echo "✅ CLI info command works"

cli-demo: ## Run a small demo evaluation
	lae-eval run --max-samples 5
	lae-eval analyze --report

# Evaluation Tasks
eval-small: ## Run small evaluation (10 samples)
	lae-eval run --max-samples 10 --chat

eval-medium: ## Run medium evaluation (100 samples)
	lae-eval run --max-samples 100 --chat --strict

eval-full: ## Run full evaluation (1000 samples)
	lae-eval run --max-samples 1000 --strict

# Analysis Tasks
analyze: ## Analyze latest results
	lae-eval analyze --plots --report --excel

analyze-all: ## Analyze all result files
	find results/ -name "evaluation_results.json" -exec lae-eval analyze --results-file {} --plots \;

# Development Scripts
run-async: ## Run high-throughput async pipeline
	python scripts/run_pipeline_async.py

inspect: ## Inspect model responses
	python scripts/inspect_model_response.py

debug: ## Debug model client
	python scripts/debug_model_client.py

# Cleanup
clean: ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-results: ## Clean up evaluation results
	rm -rf results/*/
	@echo "⚠️  All evaluation results have been deleted!"

# Build and Publish
build: ## Build distribution packages
	python -m build

publish-test: ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
	python -m twine upload dist/*

# Documentation  
docs: ## Generate documentation
	mkdir -p docs/api docs/guides docs/images
	@echo "📚 Documentation structure created"

serve-docs: ## Serve documentation locally (if using mkdocs)
	mkdocs serve

# Docker (optional)
docker-build: ## Build Docker image
	docker build -t les-audits-affaires-eval .

docker-run: ## Run evaluation in Docker
	docker run --env-file .env les-audits-affaires-eval lae-eval run --max-samples 10

# Environment Setup
setup-env: ## Create example .env file
	@if [ ! -f .env ]; then \
		echo "Creating .env file from template..."; \
		cp .env.example .env; \
		echo "✅ .env file created. Please edit with your API keys."; \
	else \
		echo "⚠️  .env file already exists"; \
	fi

check-env: ## Check environment configuration
	python -c "from les_audits_affaires_eval.config import *; print('✅ Configuration loaded successfully')"

# Performance Testing
perf-test: ## Run performance benchmarks
	python scripts/test_setup.py --benchmark

load-test: ## Run load testing
	python scripts/run_pipeline_async.py --max-samples 100 --concurrent-requests 50

# Maintenance
update-deps: ## Update dependencies
	pip-compile requirements.in
	pip-compile requirements-dev.in

security-check: ## Run security checks
	safety check
	bandit -r src/les_audits_affaires_eval/

# Git Hooks
pre-commit: ## Run pre-commit hooks manually
	pre-commit run --all-files

# Release Management
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major

# Comprehensive Quality Check
quality: format lint test ## Run all quality checks

# Full Development Cycle
dev-cycle: clean install-dev quality cli-test ## Complete development cycle check

# Production Deployment Check
prod-check: clean build test format-check lint ## Pre-production checks

# Show current configuration
config: ## Show current configuration
	@echo "📋 Current Configuration:"
	@echo "========================"
	@python -c "from les_audits_affaires_eval.config import *; print(f'Model: {MODEL_NAME}'); print(f'Endpoint: {MODEL_ENDPOINT}'); print(f'Max Samples: {MAX_SAMPLES}'); print(f'Batch Size: {BATCH_SIZE}')"

# Show project statistics
stats: ## Show project statistics
	@echo "📊 Project Statistics:"
	@echo "====================="
	@echo "Python files:"
	@find src/ -name "*.py" | wc -l
	@echo "Lines of code:"
	@find src/ -name "*.py" -exec wc -l {} + | tail -1
	@echo "Test files:"
	@find tests/ -name "*.py" | wc -l
	@echo "Scripts:"
	@find scripts/ -name "*.py" | wc -l

# New targets
test-providers:
	@echo "🔌 Testing External Provider Connections..."
	lae-eval test-providers

demo-providers:
	@echo "🎭 Running External Provider Demo..."
	python scripts/demo_external_providers.py

setup-providers:
	@echo "🔌 Setting up External Provider API Keys"
	@echo "========================================"
	@echo ""
	@echo "Add these to your ~/.bashrc or ~/.zshrc:"
	@echo ""
	@echo "# OpenAI API"
	@echo "export OPENAI_API_KEY='sk-...'"
	@echo ""
	@echo "# Mistral AI API"
	@echo "export MISTRAL_API_KEY='...'"
	@echo ""
	@echo "# Anthropic Claude API"
	@echo "export ANTHROPIC_API_KEY='sk-ant-...'"
	@echo ""
	@echo "# Google Gemini API"
	@echo "export GOOGLE_API_KEY='...'"
	@echo ""
	@echo "Then run: source ~/.bashrc"
	@echo ""
	@echo "Test with: make test-providers" 