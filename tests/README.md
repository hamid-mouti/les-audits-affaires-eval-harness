# Tests

This directory contains the test suite for the Les Audits-Affaires evaluation harness.

## Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   └── test_config.py      # Configuration function tests
├── integration/            # Integration tests for complete workflows
│   └── test_one_evaluation.py  # Single evaluation pipeline test
├── conftest.py            # Pytest fixtures and test configuration
└── README.md              # This file
```

## Running Tests

### All Tests
```bash
make test
# or
pytest tests/
```

### Unit Tests Only
```bash
make test-unit
# or
pytest tests/unit/
```

### Integration Tests Only
```bash
make test-integration
# or
pytest tests/integration/
```

## Test Types

### Unit Tests
- Test individual functions and classes in isolation
- Use mocks for external dependencies
- Fast execution, no network calls
- Located in `tests/unit/`

### Integration Tests
- Test complete workflows end-to-end
- May use real or mocked external services
- Test the interaction between components
- Located in `tests/integration/`

## Environment Variables for Testing

Integration tests can use environment variables for real API testing:

```bash
# Required for Azure OpenAI evaluation
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-key"

# Required for model evaluation
export MODEL_ENDPOINT="your-model-endpoint"

# Optional: External provider testing
export OPENAI_API_KEY="your-openai-key"
export MISTRAL_API_KEY="your-mistral-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

If environment variables are not set, integration tests will use mocks or skip gracefully.

## Adding New Tests

### Unit Tests
1. Create test files in `tests/unit/` following the pattern `test_<module>.py`
2. Import the module you're testing from `src/les_audits_affaires_eval/`
3. Use pytest fixtures from `conftest.py` for common setup
4. Mock external dependencies using `unittest.mock`

### Integration Tests
1. Create test files in `tests/integration/` following the pattern `test_<feature>.py`
2. Test complete workflows that users would actually run
3. Handle missing environment variables gracefully
4. Use appropriate timeouts for network operations

## Test Data

Test fixtures and sample data are defined in `conftest.py` and can be used across all tests. 