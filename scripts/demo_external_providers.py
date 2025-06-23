#!/usr/bin/env python3
"""
Demo script showing how external providers work (with mock responses for demonstration)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from les_audits_affaires_eval.clients.external_providers import create_client

# Test question
TEST_QUESTION = "Quelles sont les obligations légales pour créer une SARL en France?"


def demo_factory():
    """Demonstrate the factory function"""
    print("🏭 Factory Function Demo")
    print("=" * 40)

    try:
        # Show available providers
        providers = ["openai", "mistral", "claude", "gemini"]

        print("✅ Available providers:")
        for provider in providers:
            try:
                # This will fail due to missing API keys, but shows the factory works
                client_class = create_client(provider, model="test-model")
                print(f"  📦 {provider.capitalize()}: {type(client_class).__name__}")
            except ValueError as e:
                if "API key required" in str(e):
                    print(f"  📦 {provider.capitalize()}: Client class available (needs API key)")
                else:
                    print(f"  ❌ {provider.capitalize()}: {e}")

        # Test invalid provider
        try:
            invalid_client = create_client("invalid")
            print("❌ Should have failed for invalid provider")
        except ValueError as e:
            print(f"✅ Correctly rejected invalid provider: {e}")

    except Exception as e:
        print(f"❌ Factory demo failed: {e}")


def demo_client_interfaces():
    """Demonstrate client interfaces"""
    print("\n🔌 Client Interface Demo")
    print("=" * 40)

    # Show what the clients look like
    from les_audits_affaires_eval.clients.external_providers import (
        ClaudeClient,
        GeminiClient,
        MistralClient,
        OpenAIClient,
    )

    clients_info = [
        ("OpenAI GPT-4", OpenAIClient, "gpt-4o", "OpenAI API"),
        ("Mistral Large", MistralClient, "mistral-large-latest", "Mistral AI API"),
        ("Claude 3.5 Sonnet", ClaudeClient, "claude-3-5-sonnet-20241022", "Anthropic API"),
        ("Gemini 1.5 Pro", GeminiClient, "gemini-1.5-pro", "Google AI API"),
    ]

    for name, client_class, model, api_name in clients_info:
        print(f"\n📋 {name}")
        print(f"   Class: {client_class.__name__}")
        print(f"   Model: {model}")
        print(f"   API: {api_name}")
        print(f"   Methods: generate_response(), generate_response_sync()")
        print(f"   Context: async with {client_class.__name__}() as client:")


def demo_legal_prompt_format():
    """Show how legal prompts are formatted"""
    print("\n📝 Legal Prompt Format Demo")
    print("=" * 40)

    from les_audits_affaires_eval.clients.external_providers import OpenAIClient

    try:
        # Create client without API key just to show prompt formatting
        client = OpenAIClient.__new__(OpenAIClient)
        client.api_key = "demo-key"  # Fake key for demo

        # Show formatted prompt
        messages = client._format_legal_prompt(TEST_QUESTION)

        print("🎯 System Prompt:")
        print(f"   {messages[0]['content'][:100]}...")

        print("\n🎯 User Question:")
        print(f"   {messages[1]['content']}")

        print("\n✅ Required Format Elements:")
        required_sections = [
            "• Action Requise: [action] parce que [référence légale]",
            "• Délai Legal: [délai] parce que [référence légale]",
            "• Documents Obligatoires: [documents] parce que [référence légale]",
            "• Impact Financier: [coûts] parce que [référence légale]",
            "• Conséquences Non-Conformité: [risques] parce que [référence légale]",
        ]

        for section in required_sections:
            print(f"   {section}")

    except Exception as e:
        print(f"❌ Prompt format demo failed: {e}")


def demo_expected_response():
    """Show what a proper response should look like"""
    print("\n📄 Expected Response Format Demo")
    print("=" * 40)

    sample_response = """Pour créer une SARL en France, plusieurs obligations légales doivent être respectées selon le Code de commerce et les réglementations en vigueur.

La création d'une SARL nécessite le respect de formalités précises concernant la constitution, l'immatriculation et les déclarations obligatoires. Le capital social minimum, les statuts, et l'enregistrement auprès des autorités compétentes sont des étapes incontournables.

• Action Requise: Rédiger et signer les statuts de la SARL devant notaire parce que l'article L. 223-2 du Code de commerce exige un acte authentique pour la constitution

• Délai Legal: Immatriculer la société dans les 15 jours suivant la signature des statuts parce que l'article R. 123-5 du Code de commerce impose ce délai pour l'inscription au RCS  

• Documents Obligatoires: Fournir un justificatif de domiciliation et une déclaration de non-condamnation parce que l'article R. 123-54 du Code de commerce liste ces pièces obligatoires

• Impact Financier: Constituer un capital social minimum de 1 euro et payer les frais d'immatriculation de 37,45 euros parce que l'article L. 223-2 fixe le capital minimum et l'arrêté du 28 février 2020 les tarifs

• Conséquences Non-Conformité: Risque de nullité de la société et responsabilité personnelle des associés parce que l'article L. 223-1 du Code de commerce sanctionne les irrégularités de constitution"""

    print("✅ Sample Response:")
    print(sample_response)

    print("\n🔍 Analysis:")
    sections = [
        "Action Requise",
        "Délai Legal",
        "Documents Obligatoires",
        "Impact Financier",
        "Conséquences Non-Conformité",
    ]
    found_sections = [s for s in sections if s in sample_response]

    print(f"   📊 Format compliance: {len(found_sections)}/5 sections found")
    print(f"   📏 Response length: {len(sample_response)} characters")
    print(f"   ⚖️  Legal references: Contains specific Code de commerce articles")
    print(f"   🎯 Structure: Proper bullet points with 'parce que' justifications")


def main():
    """Run all demos"""
    print("🏛️ Les Audits-Affaires - External Providers Demo")
    print("=" * 60)

    print("This demo shows how the external provider clients work.")
    print("In real usage, you would set API keys as environment variables.\n")

    # Run demos
    demo_factory()
    demo_client_interfaces()
    demo_legal_prompt_format()
    demo_expected_response()

    print("\n" + "=" * 60)
    print("🚀 Getting Started")
    print("=" * 60)

    print("1. Set your API keys:")
    print("   export OPENAI_API_KEY='sk-...'")
    print("   export MISTRAL_API_KEY='...'")
    print("   export ANTHROPIC_API_KEY='sk-ant-...'")
    print("   export GOOGLE_API_KEY='...'")

    print("\n2. Test connections:")
    print("   lae-eval test-providers")

    print("\n3. Run full test:")
    print("   python scripts/test_external_providers.py")

    print("\n4. Use in your code:")
    print(
        """
   from les_audits_affaires_eval.clients import create_client
   
   async with create_client("openai", model="gpt-4o") as client:
       response = await client.generate_response("Votre question juridique...")
       print(response)
    """
    )


if __name__ == "__main__":
    main()
