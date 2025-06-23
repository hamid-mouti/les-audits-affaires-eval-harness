import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for test results"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    test_env = {
        "MODEL_ENDPOINT": "http://localhost:8000/chat",
        "MODEL_NAME": "test-model",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o-test",
        "MAX_SAMPLES": "10",
        "BATCH_SIZE": "2",
        "CONCURRENT_REQUESTS": "2",
        "TEMPERATURE": "0.1",
        "MAX_TOKENS": "1000",
    }

    # Store original values
    original_values = {}
    for key, value in test_env.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    yield test_env

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def sample_evaluation_result():
    """Sample evaluation result for testing"""
    return {
        "score_global": 75.5,
        "scores": {
            "action_requise": 80,
            "delai_legal": 72,
            "documents_obligatoires": 78,
            "impact_financier": 71,
            "consequences_non_conformite": 76,
        },
        "justifications": {
            "action_requise": "Good legal action identification",
            "delai_legal": "Adequate deadline specification",
            "documents_obligatoires": "Complete document list",
            "impact_financier": "Fair financial assessment",
            "consequences_non_conformite": "Clear consequence explanation",
        },
    }


@pytest.fixture
def sample_dataset_item():
    """Sample dataset item for testing"""
    return {
        "question": "Quelles sont les obligations légales pour créer une SARL?",
        "action_requise": "Déposer les statuts au greffe du tribunal de commerce",
        "delai_legal": "Dans les 15 jours suivant la signature des statuts",
        "documents_obligatoires": "Statuts, formulaire M0, justificatifs d'identité",
        "impact_financier": "Frais de greffe: 37,45€, capital minimum: 1€",
        "consequences_non_conformite": "Nullité de la société, responsabilité personnelle des associés",
    }
