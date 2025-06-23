#!/usr/bin/env python3
"""
Test script for external provider clients (OpenAI, Mistral, Claude, Gemini)
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from les_audits_affaires_eval.clients.external_providers import (
    ClaudeClient,
    GeminiClient,
    MistralClient,
    OpenAIClient,
    create_client,
)

# Test question
TEST_QUESTION = "Quelles sont les obligations légales pour créer une SARL en France?"


async def test_openai():
    """Test OpenAI GPT-4"""
    print("🤖 Testing OpenAI GPT-4...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set, skipping OpenAI test")
        return None

    try:
        async with OpenAIClient(api_key=api_key, model="gpt-4o") as client:
            start_time = time.time()
            response = await client.generate_response(TEST_QUESTION)
            duration = time.time() - start_time

            print(f"✅ OpenAI GPT-4 response ({duration:.2f}s):")
            print(f"📝 Length: {len(response)} characters")
            print(f"🔍 Preview: {response[:200]}...")

            # Check for required sections
            sections = [
                "Action Requise",
                "Délai Legal",
                "Documents Obligatoires",
                "Impact Financier",
                "Conséquences Non-Conformité",
            ]
            found_sections = [s for s in sections if s in response]
            print(f"📊 Found sections: {len(found_sections)}/5 - {found_sections}")

            return {
                "provider": "OpenAI",
                "duration": duration,
                "length": len(response),
                "sections": len(found_sections),
            }

    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return None


async def test_mistral():
    """Test Mistral Large"""
    print("\n🔥 Testing Mistral Large...")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not set, skipping Mistral test")
        return None

    try:
        async with MistralClient(api_key=api_key, model="mistral-large-latest") as client:
            start_time = time.time()
            response = await client.generate_response(TEST_QUESTION)
            duration = time.time() - start_time

            print(f"✅ Mistral Large response ({duration:.2f}s):")
            print(f"📝 Length: {len(response)} characters")
            print(f"🔍 Preview: {response[:200]}...")

            # Check for required sections
            sections = [
                "Action Requise",
                "Délai Legal",
                "Documents Obligatoires",
                "Impact Financier",
                "Conséquences Non-Conformité",
            ]
            found_sections = [s for s in sections if s in response]
            print(f"📊 Found sections: {len(found_sections)}/5 - {found_sections}")

            return {
                "provider": "Mistral",
                "duration": duration,
                "length": len(response),
                "sections": len(found_sections),
            }

    except Exception as e:
        print(f"❌ Mistral test failed: {e}")
        return None


async def test_claude():
    """Test Claude 3.5 Sonnet"""
    print("\n🧠 Testing Claude 3.5 Sonnet...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set, skipping Claude test")
        return None

    try:
        async with ClaudeClient(api_key=api_key, model="claude-3-5-sonnet-20241022") as client:
            start_time = time.time()
            response = await client.generate_response(TEST_QUESTION)
            duration = time.time() - start_time

            print(f"✅ Claude 3.5 Sonnet response ({duration:.2f}s):")
            print(f"📝 Length: {len(response)} characters")
            print(f"🔍 Preview: {response[:200]}...")

            # Check for required sections
            sections = [
                "Action Requise",
                "Délai Legal",
                "Documents Obligatoires",
                "Impact Financier",
                "Conséquences Non-Conformité",
            ]
            found_sections = [s for s in sections if s in response]
            print(f"📊 Found sections: {len(found_sections)}/5 - {found_sections}")

            return {
                "provider": "Claude",
                "duration": duration,
                "length": len(response),
                "sections": len(found_sections),
            }

    except Exception as e:
        print(f"❌ Claude test failed: {e}")
        return None


async def test_gemini():
    """Test Google Gemini"""
    print("\n💎 Testing Google Gemini...")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set, skipping Gemini test")
        return None

    try:
        async with GeminiClient(api_key=api_key, model="gemini-1.5-pro") as client:
            start_time = time.time()
            response = await client.generate_response(TEST_QUESTION)
            duration = time.time() - start_time

            print(f"✅ Gemini 1.5 Pro response ({duration:.2f}s):")
            print(f"📝 Length: {len(response)} characters")
            print(f"🔍 Preview: {response[:200]}...")

            # Check for required sections
            sections = [
                "Action Requise",
                "Délai Legal",
                "Documents Obligatoires",
                "Impact Financier",
                "Conséquences Non-Conformité",
            ]
            found_sections = [s for s in sections if s in response]
            print(f"📊 Found sections: {len(found_sections)}/5 - {found_sections}")

            return {
                "provider": "Gemini",
                "duration": duration,
                "length": len(response),
                "sections": len(found_sections),
            }

    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return None


def test_factory():
    """Test the factory function"""
    print("\n🏭 Testing factory function...")

    try:
        # Test valid providers
        openai_client = create_client("openai", model="gpt-4o")
        mistral_client = create_client("mistral", model="mistral-large-latest")
        claude_client = create_client("claude", model="claude-3-5-sonnet-20241022")
        gemini_client = create_client("gemini", model="gemini-1.5-pro")

        print(f"✅ Factory created: {type(openai_client).__name__}")
        print(f"✅ Factory created: {type(mistral_client).__name__}")
        print(f"✅ Factory created: {type(claude_client).__name__}")
        print(f"✅ Factory created: {type(gemini_client).__name__}")

        # Test invalid provider
        try:
            invalid_client = create_client("invalid")
            print("❌ Factory should have failed for invalid provider")
        except ValueError as e:
            print(f"✅ Factory correctly rejected invalid provider: {e}")

    except Exception as e:
        print(f"❌ Factory test failed: {e}")


async def main():
    """Run all tests"""
    print("🏛️ Les Audits-Affaires - External Provider Tests")
    print("=" * 60)

    print(f"🔍 Test question: {TEST_QUESTION}")
    print("=" * 60)

    # Test factory first
    test_factory()

    # Run API tests
    results = []

    # Test each provider
    tasks = [
        test_openai(),
        test_mistral(),
        test_claude(),
        test_gemini(),
    ]

    for task in tasks:
        result = await task
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    if results:
        print("🎯 Provider Performance:")
        for result in results:
            print(
                f"  {result['provider']:12} | {result['duration']:6.2f}s | {result['length']:5d} chars | {result['sections']}/5 sections"
            )

        # Best performer
        fastest = min(results, key=lambda x: x["duration"])
        most_structured = max(results, key=lambda x: x["sections"])
        longest = max(results, key=lambda x: x["length"])

        print(f"\n🏆 Fastest: {fastest['provider']} ({fastest['duration']:.2f}s)")
        print(
            f"📋 Most Structured: {most_structured['provider']} ({most_structured['sections']}/5 sections)"
        )
        print(f"📝 Most Detailed: {longest['provider']} ({longest['length']} characters)")

    else:
        print("❌ No successful tests - check your API keys!")

    print("\n💡 To set API keys:")
    print("  export OPENAI_API_KEY='your-openai-key'")
    print("  export MISTRAL_API_KEY='your-mistral-key'")
    print("  export ANTHROPIC_API_KEY='your-anthropic-key'")
    print("  export GOOGLE_API_KEY='your-google-key'")


if __name__ == "__main__":
    asyncio.run(main())
