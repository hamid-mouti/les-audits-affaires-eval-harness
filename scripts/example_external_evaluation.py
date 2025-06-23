#!/usr/bin/env python3
"""
Example: Using external providers for Les Audits-Affaires evaluation
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset

from les_audits_affaires_eval.clients import EvaluatorClient, create_client


async def evaluate_with_external_provider(provider: str, model: str, max_samples: int = 5):
    """Evaluate using an external provider"""
    print(f"🔍 Evaluating with {provider.upper()} ({model})")
    print("=" * 50)

    # Load sample data
    try:
        dataset = load_dataset("legmlai/les-audits-affaires", split="train")
        samples = list(dataset)
        if max_samples:
            samples = samples[:max_samples]
        print(f"📊 Loaded {len(samples)} samples")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Create clients
    try:
        model_client = create_client(provider, model=model)
        evaluator_client = EvaluatorClient()
    except Exception as e:
        print(f"❌ Failed to create clients: {e}")
        return

    results = []

    try:
        async with model_client as client:
            for i, sample in enumerate(samples):
                print(f"\n📝 Sample {i+1}/{len(samples)}: {sample['question'][:80]}...")

                try:
                    # Generate response
                    response = await client.generate_response(sample["question"])
                    print(f"✅ Response length: {len(response)} characters")

                    # Evaluate response
                    evaluation = evaluator_client.evaluate_response(
                        sample["question"], response, sample
                    )

                    result = {
                        "sample_id": i,
                        "question": sample["question"],
                        "response": response,
                        "evaluation": evaluation,
                        "provider": provider,
                        "model": model,
                    }
                    results.append(result)

                    print(f"📊 Score: {evaluation.get('score_global', 0):.2f}")

                except Exception as e:
                    print(f"❌ Error processing sample {i+1}: {e}")
                    continue

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return

    # Calculate summary statistics
    if results:
        scores = [r["evaluation"].get("score_global", 0) for r in results]
        avg_score = sum(scores) / len(scores)

        print(f"\n🎯 Results Summary:")
        print(f"   Provider: {provider.upper()}")
        print(f"   Model: {model}")
        print(f"   Samples: {len(results)}")
        print(f"   Average Score: {avg_score:.2f}")
        print(f"   Score Range: {min(scores):.2f} - {max(scores):.2f}")

        # Save results
        output_file = f"results_{provider}_{model.replace('/', '_')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to: {output_file}")

    return results


async def compare_providers(providers_config: Dict[str, Dict[str, str]], max_samples: int = 3):
    """Compare multiple providers"""
    print("🏆 Provider Comparison")
    print("=" * 60)

    all_results = {}

    for provider_name, config in providers_config.items():
        print(f"\n🔄 Testing {provider_name}...")

        # Check if API key is available
        api_key_vars = {
            "openai": "OPENAI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }

        api_key_var = api_key_vars.get(provider_name.lower())
        if not api_key_var or not os.getenv(api_key_var):
            print(f"⏭️  Skipping {provider_name} - no API key ({api_key_var})")
            continue

        try:
            results = await evaluate_with_external_provider(
                provider_name, config["model"], max_samples
            )
            if results:
                all_results[provider_name] = results
        except Exception as e:
            print(f"❌ {provider_name} failed: {e}")

    # Summary comparison
    if all_results:
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 60)

        for provider, results in all_results.items():
            scores = [r["evaluation"].get("score_global", 0) for r in results]
            avg_score = sum(scores) / len(scores)
            print(f"{provider:15} | {avg_score:6.2f} avg | {len(results):3d} samples")

        # Best performer
        best_provider = max(
            all_results.keys(),
            key=lambda p: sum(r["evaluation"].get("score_global", 0) for r in all_results[p])
            / len(all_results[p]),
        )

        best_avg = sum(
            r["evaluation"].get("score_global", 0) for r in all_results[best_provider]
        ) / len(all_results[best_provider])

        print(f"\n🏆 Best Performer: {best_provider} ({best_avg:.2f} average score)")
    else:
        print("❌ No successful evaluations - check your API keys!")


async def main():
    """Main function"""
    print("🏛️ Les Audits-Affaires - External Provider Evaluation")
    print("=" * 60)

    # Configuration for providers to test
    providers_config = {
        "openai": {"model": "gpt-4o"},
        "mistral": {"model": "mistral-large-latest"},
        "claude": {"model": "claude-3-5-sonnet-20241022"},
        "gemini": {"model": "gemini-1.5-pro"},
    }

    # Check what's available
    available_providers = []
    api_key_vars = {
        "openai": "OPENAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }

    for provider in providers_config.keys():
        api_key_var = api_key_vars.get(provider)
        if api_key_var and os.getenv(api_key_var):
            available_providers.append(provider)

    if not available_providers:
        print("❌ No API keys found! Set up your keys first:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export MISTRAL_API_KEY='...'")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        print("   export GOOGLE_API_KEY='...'")
        print("\nOr run: make setup-providers")
        return

    print(f"✅ Available providers: {', '.join(available_providers)}")

    # Run comparison
    await compare_providers(providers_config, max_samples=3)

    print(f"\n💡 Tips:")
    print(f"   • Increase max_samples for more comprehensive evaluation")
    print(f"   • Results are saved as JSON files for further analysis")
    print(f"   • Use lae-eval analyze to generate plots and reports")


if __name__ == "__main__":
    asyncio.run(main())
