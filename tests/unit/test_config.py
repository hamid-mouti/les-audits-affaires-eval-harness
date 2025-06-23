import os

import pytest

from les_audits_affaires_eval.config import (
    BATCH_SIZE,
    MAX_SAMPLES,
    MODEL_ENDPOINT,
    MODEL_NAME,
    get_safe_model_name,
)


def test_get_safe_model_name():
    """Test model name sanitization"""
    # Test basic sanitization
    assert get_safe_model_name("model/name-v1.0") == "model_name_v1_0"
    assert get_safe_model_name("openai:gpt-4") == "openai_gpt_4"
    assert get_safe_model_name("simple_name") == "simple_name"

    # Test edge cases
    assert get_safe_model_name("") == ""
    assert get_safe_model_name("model//name") == "model__name"


def test_config_defaults():
    """Test that config has reasonable defaults"""
    assert isinstance(MAX_SAMPLES, int)
    assert isinstance(BATCH_SIZE, int)
    assert MAX_SAMPLES > 0
    assert BATCH_SIZE > 0


def test_config_with_env_vars(mock_env_vars):
    """Test config loading with environment variables"""
    # Re-import to get updated config
    import importlib

    from les_audits_affaires_eval import config

    importlib.reload(config)

    assert config.MODEL_ENDPOINT == "http://localhost:8000/chat"
    assert config.MODEL_NAME == "test-model"
    assert config.MAX_SAMPLES == 10
    assert config.BATCH_SIZE == 2
