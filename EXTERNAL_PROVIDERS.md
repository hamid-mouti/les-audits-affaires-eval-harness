# External Provider Support

The Les Audits-Affaires evaluation harness now supports popular external LLM providers for easy benchmarking and comparison.

## Supported Providers

| Provider | Models | API Documentation |
|----------|--------|-------------------|
| **OpenAI** | `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo` | [OpenAI API](https://platform.openai.com/docs) |
| **Mistral** | `mistral-large-latest`, `mistral-medium-latest` | [Mistral API](https://docs.mistral.ai/) |
| **Claude** | `claude-3-5-sonnet-20241022`, `claude-3-haiku` | [Anthropic API](https://docs.anthropic.com/) |
| **Gemini** | `gemini-1.5-pro`, `gemini-1.0-pro` | [Google AI API](https://ai.google.dev/) |

## Quick Start

### 1. Set up API Keys

```bash
# OpenAI
export OPENAI_API_KEY='sk-...'

# Mistral AI
export MISTRAL_API_KEY='...'

# Anthropic Claude
export ANTHROPIC_API_KEY='sk-ant-...'

# Google Gemini
export GOOGLE_API_KEY='...'
```

Or use the helper:
```bash
make setup-providers
```

### 2. Test Connections

```bash
# Test which providers are available
lae-eval test-providers

# Or using make
make test-providers
```

### 3. Run Demo

```bash
# See how providers work
make demo-providers

# Run example evaluation
python scripts/example_external_evaluation.py
```

## Usage Examples

### Basic Usage

```python
from les_audits_affaires_eval.clients import create_client

# Create a client
async with create_client("openai", model="gpt-4o") as client:
    response = await client.generate_response("Votre question juridique...")
    print(response)
```

### Factory Pattern

```python
from les_audits_affaires_eval.clients import create_client

# Available providers: openai, mistral, claude, gemini
providers = {
    "openai": {"model": "gpt-4o"},
    "mistral": {"model": "mistral-large-latest"},
    "claude": {"model": "claude-3-5-sonnet-20241022"},
    "gemini": {"model": "gemini-1.5-pro"},
}

for provider_name, config in providers.items():
    client = create_client(provider_name, **config)
    # Use client...
```

### Full Evaluation

```python
import asyncio
from les_audits_affaires_eval.clients import create_client, EvaluatorClient
from datasets import load_dataset

async def evaluate_provider(provider: str, model: str):
    # Load data
    dataset = load_dataset("legmlai/les-audits-affaires", split="train")
    samples = list(dataset)[:5]  # Test with 5 samples
    
    # Create clients
    model_client = create_client(provider, model=model)
    evaluator = EvaluatorClient()
    
    results = []
    async with model_client as client:
        for sample in samples:
            # Generate response
            response = await client.generate_response(sample['question'])
            
            # Evaluate
            evaluation = evaluator.evaluate_response(
                sample['question'], response, sample
            )
            
            results.append({
                'question': sample['question'],
                'response': response,
                'evaluation': evaluation
            })
    
    return results

# Run evaluation
results = asyncio.run(evaluate_provider("openai", "gpt-4o"))
```

## Legal Prompt Format

All external providers use the same specialized legal prompt format:

```
Tu es un expert juridique français spécialisé en droit des affaires et droit commercial.

Réponds à la question juridique ci-dessous, puis termine ta réponse par un résumé structuré avec ces 5 éléments:

• Action Requise: [décris l'action concrète nécessaire] parce que [référence légale précise]
• Délai Legal: [indique le délai précis] parce que [référence légale précise]  
• Documents Obligatoires: [liste les documents nécessaires] parce que [référence légale précise]
• Impact Financier: [estime les coûts/frais] parce que [référence légale précise]
• Conséquences Non-Conformité: [explique les risques] parce que [référence légale précise]

Question: {question}
```

## Expected Response Format

```
Pour créer une SARL en France, plusieurs obligations légales doivent être respectées...

• Action Requise: Rédiger et signer les statuts de la SARL devant notaire parce que l'article L. 223-2 du Code de commerce exige un acte authentique pour la constitution

• Délai Legal: Immatriculer la société dans les 15 jours suivant la signature des statuts parce que l'article R. 123-5 du Code de commerce impose ce délai pour l'inscription au RCS  

• Documents Obligatoires: Fournir un justificatif de domiciliation et une déclaration de non-condamnation parce que l'article R. 123-54 du Code de commerce liste ces pièces obligatoires

• Impact Financier: Constituer un capital social minimum de 1 euro et payer les frais d'immatriculation de 37,45 euros parce que l'article L. 223-2 fixe le capital minimum et l'arrêté du 28 février 2020 les tarifs

• Conséquences Non-Conformité: Risque de nullité de la société et responsabilité personnelle des associés parce que l'article L. 223-1 du Code de commerce sanctionne les irrégularités de constitution
```

## CLI Integration

The external providers are fully integrated into the CLI:

```bash
# Show provider status
lae-eval info

# Test connections
lae-eval test-providers

# The main evaluation commands work with your existing setup
lae-eval run --chat --max-samples 100
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `scripts/test_external_providers.py` | Comprehensive testing of all providers |
| `scripts/demo_external_providers.py` | Demo functionality without API calls |
| `scripts/example_external_evaluation.py` | Full evaluation example |

## Makefile Targets

```bash
make test-providers     # Test provider connections
make demo-providers     # Run demo
make setup-providers    # Show setup instructions
```

## Architecture

```
src/les_audits_affaires_eval/clients/
├── __init__.py                 # Exports all clients
├── external_providers.py       # External provider implementations
└── (existing model_client.py)  # Original local model clients
```

Each provider client implements:
- `async generate_response(question: str) -> str`
- `generate_response_sync(question: str) -> str`
- Async context manager support
- Retry logic with exponential backoff
- Proper error handling

## Error Handling

- Missing API keys are handled gracefully
- Network errors trigger automatic retries
- Invalid responses are caught and logged
- Providers can be tested individually

## Performance Considerations

- Async/await for concurrent requests
- Connection pooling via aiohttp
- Configurable timeouts (5 minutes default)
- Retry logic with exponential backoff
- Rate limiting respect (provider-dependent)

## Future Enhancements

- [ ] Support for more providers (Cohere, Together, etc.)
- [ ] Custom model configurations
- [ ] Batch processing optimization
- [ ] Cost tracking and reporting
- [ ] Provider-specific optimizations 