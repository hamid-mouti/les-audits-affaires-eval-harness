# Scripts Directory

This directory contains utility scripts for the Les Audits-Affaires evaluation harness.

## Available Scripts

| Script | Description | Usage |
|--------|-------------|--------|
| `test_external_providers.py` | Test all external provider connections with real API calls | `python scripts/test_external_providers.py` |
| `demo_external_providers.py` | Demo external provider functionality without API calls | `python scripts/demo_external_providers.py` |
| `example_external_evaluation.py` | Complete evaluation example using external providers | `python scripts/example_external_evaluation.py` |

## Makefile Shortcuts

```bash
make test-providers    # Run test_external_providers.py
make demo-providers    # Run demo_external_providers.py
```

## Requirements

- External provider scripts require API keys set as environment variables
- See `.env.example` for required environment variables
- Use `make setup-providers` for setup instructions 