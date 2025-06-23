"""
External provider clients for popular LLM APIs
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for OpenAI API (GPT-4, GPT-3.5, etc.)"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.async_client = None

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

    async def __aenter__(self):
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.async_client:
            await self.async_client.close()

    def _format_legal_prompt(self, question: str) -> List[Dict[str, str]]:
        """Format question for legal analysis"""
        system_prompt = """Tu es un expert juridique français spécialisé en droit des affaires et droit commercial.

Réponds à la question juridique ci-dessous, puis termine ta réponse par un résumé structuré avec ces 5 éléments:

• Action Requise: [décris l'action concrète nécessaire] parce que [référence légale précise]
• Délai Legal: [indique le délai précis] parce que [référence légale précise]  
• Documents Obligatoires: [liste les documents nécessaires] parce que [référence légale précise]
• Impact Financier: [estime les coûts/frais] parce que [référence légale précise]
• Conséquences Non-Conformité: [explique les risques] parce que [référence légale précise]

Utilise des références légales différentes pour chaque catégorie et sois précis dans tes explications."""

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, question: str) -> str:
        """Generate response using OpenAI API"""
        if not self.async_client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        messages = self._format_legal_prompt(question)

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.1, max_tokens=4000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def generate_response_sync(self, question: str) -> str:
        """Synchronous version"""
        if not self.client:
            self.client = OpenAI(api_key=self.api_key)

        messages = self._format_legal_prompt(question)

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.1, max_tokens=4000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class MistralClient:
    """Client for Mistral AI API"""

    def __init__(self, api_key: str = None, model: str = "mistral-large-latest"):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.endpoint = "https://api.mistral.ai/v1/chat/completions"
        self.session = None

        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable.")

    async def __aenter__(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _format_legal_prompt(self, question: str) -> List[Dict[str, str]]:
        """Format question for legal analysis"""
        system_prompt = """Tu es un expert juridique français spécialisé en droit des affaires et droit commercial.

Réponds à la question juridique ci-dessous, puis termine ta réponse par un résumé structuré avec ces 5 éléments:

• Action Requise: [décris l'action concrète nécessaire] parce que [référence légale précise]
• Délai Legal: [indique le délai précis] parce que [référence légale précise]  
• Documents Obligatoires: [liste les documents nécessaires] parce que [référence légale précise]
• Impact Financier: [estime les coûts/frais] parce que [référence légale précise]
• Conséquences Non-Conformité: [explique les risques] parce que [référence légale précise]"""

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, question: str) -> str:
        """Generate response using Mistral API"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        messages = self._format_legal_prompt(question)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4000,
        }

        try:
            async with self.session.post(self.endpoint, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Mistral API error {response.status}: {error_text}")
                    raise Exception(f"Mistral API error {response.status}: {error_text}")

                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            raise


class ClaudeClient:
    """Client for Anthropic Claude API"""

    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.endpoint = "https://api.anthropic.com/v1/messages"
        self.session = None

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )

    async def __aenter__(self):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _format_legal_prompt(self, question: str) -> str:
        """Format question for legal analysis"""
        return f"""Tu es un expert juridique français spécialisé en droit des affaires et droit commercial.

Réponds à la question juridique ci-dessous, puis termine ta réponse par un résumé structuré avec ces 5 éléments:

• Action Requise: [décris l'action concrète nécessaire] parce que [référence légale précise]
• Délai Legal: [indique le délai précis] parce que [référence légale précise]  
• Documents Obligatoires: [liste les documents nécessaires] parce que [référence légale précise]
• Impact Financier: [estime les coûts/frais] parce que [référence légale précise]
• Conséquences Non-Conformité: [explique les risques] parce que [référence légale précise]

Question: {question}"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, question: str) -> str:
        """Generate response using Claude API"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        prompt = self._format_legal_prompt(question)

        payload = {
            "model": self.model,
            "max_tokens": 4000,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            async with self.session.post(self.endpoint, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Claude API error {response.status}: {error_text}")
                    raise Exception(f"Claude API error {response.status}: {error_text}")

                result = await response.json()
                return result["content"][0]["text"].strip()

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


class GeminiClient:
    """Client for Google Gemini API"""

    def __init__(self, api_key: str = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.session = None

        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")

        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _format_legal_prompt(self, question: str) -> str:
        """Format question for legal analysis"""
        return f"""Tu es un expert juridique français spécialisé en droit des affaires et droit commercial.

Réponds à la question juridique ci-dessous, puis termine ta réponse par un résumé structuré avec ces 5 éléments:

• Action Requise: [décris l'action concrète nécessaire] parce que [référence légale précise]
• Délai Legal: [indique le délai précis] parce que [référence légale précise]  
• Documents Obligatoires: [liste les documents nécessaires] parce que [référence légale précise]
• Impact Financier: [estime les coûts/frais] parce que [référence légale précise]
• Conséquences Non-Conformité: [explique les risques] parce que [référence légale précise]

Question: {question}"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, question: str) -> str:
        """Generate response using Gemini API"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        prompt = self._format_legal_prompt(question)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4000},
        }

        params = {"key": self.api_key}

        try:
            async with self.session.post(self.endpoint, json=payload, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Gemini API error {response.status}: {error_text}")
                    raise Exception(f"Gemini API error {response.status}: {error_text}")

                result = await response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


# Factory function for easy client creation
def create_client(provider: str, **kwargs):
    """Factory function to create clients for different providers"""
    providers = {
        "openai": OpenAIClient,
        "mistral": MistralClient,
        "claude": ClaudeClient,
        "gemini": GeminiClient,
    }

    if provider.lower() not in providers:
        raise ValueError(f"Unsupported provider: {provider}. Available: {list(providers.keys())}")

    return providers[provider.lower()](**kwargs)
