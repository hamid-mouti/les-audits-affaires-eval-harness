# Scripts Directory

This directory contains utility scripts for the Les Audits-Affaires evaluation harness.

## Available Scripts

| Script | Description | Usage |
|--------|-------------|--------|
| `test_external_providers.py` | Test all external provider connections with real API calls | `python scripts/test_external_providers.py` |
| `demo_external_providers.py` | Demo external provider functionality without API calls | `python scripts/demo_external_providers.py` |
| `example_external_evaluation.py` | Complete evaluation example using external providers | `python scripts/example_external_evaluation.py` |
| `upload_results.py` | Upload evaluation results to HuggingFace leaderboard dataset | `python scripts/upload_results.py --model_name gpt-4o --provider openai` |
| `quick_upload.py` | Quick upload with auto-detection of result files | `python scripts/quick_upload.py gpt-4o openai` |
| `batch_evaluate_and_upload.py` | Batch process all pending requests, evaluate and upload results | `python scripts/batch_evaluate_and_upload.py --simulate` |

## Results Upload Scripts

### Quick Upload (Recommended)
```bash
# Auto-detect and upload results for a model
python scripts/quick_upload.py gpt-4o openai
python scripts/quick_upload.py claude-3 anthropic
python scripts/quick_upload.py llama-3-8b meta
```

### Manual Upload
```bash
# Upload specific results file
python scripts/upload_results.py --model_name gpt-4o --provider openai --results_file results_openai_gpt-4o.json

# Upload from results directory
python scripts/upload_results.py --model_name gpt-4o --provider openai --results_dir results/gpt_4o/

# With custom request ID
python scripts/upload_results.py --model_name gpt-4o --provider openai --request_id req_abc123
```

### Batch Processing
```bash
# Process all pending requests with simulation (safe for testing)
python scripts/batch_evaluate_and_upload.py --simulate --dry-run

# Process all pending requests (real evaluation required)
python scripts/batch_evaluate_and_upload.py

# Process maximum 5 requests
python scripts/batch_evaluate_and_upload.py --max-requests 5

# Clear all requests and start fresh (DANGEROUS!)
python scripts/batch_evaluate_and_upload.py --clear-requests --dry-run
```

### Environment Setup for Uploads
```bash
# Set HuggingFace token for dataset uploads
export HF_TOKEN="your_huggingface_token"

# Or create .env file with:
echo "HF_TOKEN=your_huggingface_token" >> .env
```

### Workflow for Automated Processing

1. **Run evaluations** and save results to `/results` directory
2. **Upload individual results**: `python scripts/quick_upload.py model-name provider`
3. **Batch process pending requests**: `python scripts/batch_evaluate_and_upload.py --simulate`

## Makefile Shortcuts

```bash
make test-providers    # Run test_external_providers.py
make demo-providers    # Run demo_external_providers.py
```

## Requirements

- External provider scripts require API keys set as environment variables
- See `.env.example` for required environment variables
- Use `make setup-providers` for setup instructions 