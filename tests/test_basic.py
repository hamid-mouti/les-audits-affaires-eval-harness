#!/usr/bin/env python3
"""
Basic unit tests for Les Audits-Affaires Evaluation Harness
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBasicImports(unittest.TestCase):
    """Test basic imports and package structure"""

    def test_package_imports(self):
        """Test that all main package components import correctly"""
        try:
            from src.les_audits_affaires_eval import (
                EvaluatorClient,
                LesAuditsAffairesEvaluator,
                ModelClient,
                load_evaluation_results,
            )

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Package import failed: {e}")

    def test_config_import(self):
        """Test configuration import"""
        try:
            from src.les_audits_affaires_eval.config import (
                AZURE_OPENAI_ENDPOINT,
                DATASET_NAME,
                MODEL_ENDPOINT,
            )

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Config import failed: {e}")


class TestModelClient(unittest.TestCase):
    """Test ModelClient class"""

    def setUp(self):
        from src.les_audits_affaires_eval.model_client import ModelClient

        self.client = ModelClient()

    def test_format_prompt(self):
        """Test prompt formatting"""
        question = "Test question"
        prompt = self.client._format_prompt(question)

        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_end|>", prompt)
        self.assertIn("Test question", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    def test_client_initialization(self):
        """Test client initialization"""
        self.assertIsNotNone(self.client.endpoint)
        self.assertIsNotNone(self.client.model_name)


class TestEvaluatorClient(unittest.TestCase):
    """Test EvaluatorClient class"""

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://test.cognitiveservices.azure.com/",
            "AZURE_OPENAI_API_KEY": "test_key",
        },
    )
    def test_evaluator_initialization(self):
        """Test evaluator client initialization"""
        try:
            from src.les_audits_affaires_eval.model_client import EvaluatorClient

            client = EvaluatorClient()
            self.assertIsNotNone(client.client)
        except Exception as e:
            # This might fail due to Azure OpenAI setup, but we test the import
            self.assertIn("EvaluatorClient", str(type(e).__name__) or "")

    def test_default_evaluation_structure(self):
        """Test default evaluation response structure"""
        from src.les_audits_affaires_eval.model_client import EvaluatorClient

        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://test.cognitiveservices.azure.com/",
                "AZURE_OPENAI_API_KEY": "test_key",
            },
        ):
            client = EvaluatorClient()
            default_eval = client._create_default_evaluation()

            self.assertIn("score_global", default_eval)
            self.assertIn("scores", default_eval)
            self.assertIn("justifications", default_eval)

            # Check all required categories
            required_categories = [
                "action_requise",
                "delai_legal",
                "documents_obligatoires",
                "impact_financier",
                "consequences_non_conformite",
            ]
            for category in required_categories:
                self.assertIn(category, default_eval["scores"])
                self.assertIn(category, default_eval["justifications"])


class TestEvaluator(unittest.TestCase):
    """Test main evaluator class"""

    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        from src.les_audits_affaires_eval.evaluator import LesAuditsAffairesEvaluator

        evaluator = LesAuditsAffairesEvaluator()
        self.assertIsNotNone(evaluator)

    @patch("src.les_audits_affaires_eval.evaluator.load_dataset")
    def test_load_dataset(self, mock_load):
        """Test dataset loading"""
        from src.les_audits_affaires_eval.evaluator import LesAuditsAffairesEvaluator

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load.return_value = mock_dataset

        evaluator = LesAuditsAffairesEvaluator()
        dataset = evaluator._load_dataset(10)

        self.assertIsNotNone(dataset)
        mock_load.assert_called_once()


class TestUtils(unittest.TestCase):
    """Test utility functions"""

    def test_utils_import(self):
        """Test utils import"""
        try:
            from src.les_audits_affaires_eval.utils import (
                create_score_distribution_plot,
                generate_analysis_report,
                load_evaluation_results,
            )

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Utils import failed: {e}")

    def test_results_directory_creation(self):
        """Test results directory handling"""
        from src.les_audits_affaires_eval.config import RESULTS_DIR

        # Should be a string path
        self.assertIsInstance(RESULTS_DIR, str)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
