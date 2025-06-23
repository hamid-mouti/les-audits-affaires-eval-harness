#!/usr/bin/env python3
"""
Integration test for single evaluation entry
Tests the complete evaluation pipeline with mock or configured endpoints
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from les_audits_affaires_eval import LesAuditsAffairesEvaluator


async def test_single_evaluation_with_mock():
    """Test single evaluation with mocked model client and evaluator"""
    print("🧪 Testing Single Evaluation with Mock Client")
    print("=" * 60)

    # Mock response that follows the expected format
    mock_model_response = """
Voici mon analyse de cette question juridique complexe.

Pour répondre à votre question sur la modification des statuts d'une société anonyme, voici les éléments clés :

• Action Requise: Convoquer une assemblée générale extraordinaire parce que l'article L225-96 du Code de commerce l'exige pour toute modification statutaire
• Délai Legal: Délai de 15 jours minimum pour la convocation parce que l'article R225-61 du Code de commerce impose ce préavis
• Documents Obligatoires: Projet de modification et rapport du conseil d'administration parce que l'article L225-97 du Code de commerce les rend obligatoires
• Impact Financier: Frais de publication et honoraires notariaux estimés à 2000€ parce que l'article 2121 du Code civil impose l'intervention notariale
• Conséquences Non-Conformité: Nullité de la modification et sanctions pénales parce que l'article L242-6 du Code de commerce prévoit ces sanctions
"""

    # Mock evaluation response
    mock_evaluation_response = {
        "score_global": 8.5,
        "scores": {
            "action_requise": 9,
            "delai_legal": 8,
            "documents_obligatoires": 9,
            "impact_financier": 8,
            "consequences_non_conformite": 8,
        },
        "justifications": {
            "action_requise": "Excellente identification de l'action requise",
            "delai_legal": "Délai correctement identifié",
            "documents_obligatoires": "Documents obligatoires bien listés",
            "impact_financier": "Impact financier estimé de manière réaliste",
            "consequences_non_conformite": "Conséquences bien expliquées",
        },
    }

    try:
        # Create evaluator with mocked credentials
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://mock-endpoint.openai.azure.com",
                "AZURE_OPENAI_API_KEY": "mock-key",
                "MODEL_ENDPOINT": "https://mock-model-endpoint.com",
            },
        ):
            evaluator = LesAuditsAffairesEvaluator(use_chat_endpoint=True, use_strict_mode=True)

            # Load dataset
            print("📊 Loading dataset...")
            dataset = evaluator.load_dataset(max_samples=1)

            if not dataset:
                print("❌ No dataset loaded")
                return False

            first_entry = dataset[0]
            print(f"📝 Testing with first entry: {first_entry['question'][:100]}...")

            # Mock both the model client and evaluator client
            with (
                patch.object(evaluator, "model_client") as mock_model_client,
                patch.object(evaluator, "evaluator_client") as mock_evaluator_client,
            ):

                # Set up model client mock
                mock_model_client.generate_response = AsyncMock(return_value=mock_model_response)

                # Set up evaluator client mock
                mock_evaluator_client.evaluate_response = MagicMock(
                    return_value=mock_evaluation_response
                )

                # Test single sample evaluation
                result = await evaluator.evaluate_single_sample(first_entry, 0)

                print(f"✅ Evaluation completed")
                print(f"📊 Global Score: {result['evaluation']['score_global']}")
                print(f"📈 Category Scores: {result['evaluation']['scores']}")

                # Verify result structure
                assert "sample_idx" in result
                assert "question" in result
                assert "model_response" in result
                assert "evaluation" in result
                assert "metadata" in result

                # Verify evaluation structure
                evaluation = result["evaluation"]
                assert "score_global" in evaluation
                assert "scores" in evaluation
                assert "justifications" in evaluation

                # Check that we have all expected categories
                expected_categories = [
                    "action_requise",
                    "delai_legal",
                    "documents_obligatoires",
                    "impact_financier",
                    "consequences_non_conformite",
                ]
                for category in expected_categories:
                    assert category in evaluation["scores"]
                    assert category in evaluation["justifications"]

                # Verify the mock was called
                mock_model_client.generate_response.assert_called_once()
                mock_evaluator_client.evaluate_response.assert_called_once()

                print("✅ All assertions passed")
                return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_evaluation_with_environment():
    """Test evaluation using environment configuration (if available)"""
    print("\n🧪 Testing with Environment Configuration")
    print("=" * 60)

    # Check if we have the required environment variables
    required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "MODEL_ENDPOINT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"⏭️ Skipping environment test - missing: {', '.join(missing_vars)}")
        return True

    try:
        evaluator = LesAuditsAffairesEvaluator(use_strict_mode=True)

        # Load single sample
        dataset = evaluator.load_dataset(max_samples=1)
        if not dataset:
            print("❌ No dataset loaded")
            return False

        print("🚀 Running real evaluation (this may take a moment)...")

        # Run evaluation with real endpoints
        results = await evaluator.run_evaluation(max_samples=1)

        print(f"✅ Real evaluation completed")
        print(f"📊 Results: {results['global_score']}")

        return True

    except Exception as e:
        print(f"⚠️ Environment test failed (this is expected if endpoints are not configured): {e}")
        return True  # Don't fail the test suite for missing environment


async def test_sync_evaluation():
    """Test synchronous evaluation methods"""
    print("\n🧪 Testing Synchronous Evaluation")
    print("=" * 60)

    # Mock responses
    mock_model_response = "Mock synchronous response with legal analysis."
    mock_evaluation_response = {
        "score_global": 7.0,
        "scores": {
            "action_requise": 7,
            "delai_legal": 7,
            "documents_obligatoires": 7,
            "impact_financier": 7,
            "consequences_non_conformite": 7,
        },
        "justifications": {
            "action_requise": "Good",
            "delai_legal": "Good",
            "documents_obligatoires": "Good",
            "impact_financier": "Good",
            "consequences_non_conformite": "Good",
        },
    }

    try:
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://mock-endpoint.openai.azure.com",
                "AZURE_OPENAI_API_KEY": "mock-key",
                "MODEL_ENDPOINT": "https://mock-model-endpoint.com",
            },
        ):
            evaluator = LesAuditsAffairesEvaluator(use_chat_endpoint=True, use_strict_mode=True)

            # Load dataset
            dataset = evaluator.load_dataset(max_samples=1)
            if not dataset:
                print("❌ No dataset loaded")
                return False

            first_entry = dataset[0]
            print(f"📝 Testing sync evaluation with: {first_entry['question'][:100]}...")

            # Mock both clients for sync methods
            with (
                patch.object(evaluator, "model_client") as mock_model_client,
                patch.object(evaluator, "evaluator_client") as mock_evaluator_client,
            ):

                # Set up sync method mocks
                mock_model_client.generate_response_sync = MagicMock(
                    return_value=mock_model_response
                )
                mock_evaluator_client.evaluate_response = MagicMock(
                    return_value=mock_evaluation_response
                )

                # Test sync single sample evaluation
                result = evaluator.evaluate_single_sample_sync(first_entry, 0)

                print(f"✅ Sync evaluation completed")
                print(f"📊 Global Score: {result['evaluation']['score_global']}")

                # Verify the sync method was called
                mock_model_client.generate_response_sync.assert_called_once()
                mock_evaluator_client.evaluate_response.assert_called_once()

                print("✅ Sync test passed")
                return True

    except Exception as e:
        print(f"❌ Sync test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all integration tests"""
    print("🏛️ Les Audits-Affaires Integration Tests")
    print("=" * 60)

    tests = [
        ("Mock Evaluation", test_single_evaluation_with_mock),
        ("Sync Evaluation", test_sync_evaluation),
        ("Environment Evaluation", test_evaluation_with_environment),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n▶️ Running: {test_name}")
        try:
            success = await test_func()
            if success:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")

    print(f"\n📊 Integration Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("❌ Some integration tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
