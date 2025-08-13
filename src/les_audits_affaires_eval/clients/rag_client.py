
import asyncio
import json
import logging
import os

import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class RAGClient:
    """Client for an external RAG pipeline with token-based authentication."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.endpoint = os.getenv("MODEL_ENDPOINT")
        self.token_endpoint = os.getenv("RAG_TOKEN_ENDPOINT")
        self.username = os.getenv("RAG_USERNAME")
        self.password = os.getenv("RAG_PASSWORD")

        if not self.endpoint or not self.token_endpoint or not self.username or not self.password:
            raise ValueError(
                "MODEL_ENDPOINT, RAG_TOKEN_ENDPOINT, RAG_USERNAME, and RAG_PASSWORD environment variables must be set."
            )

        self.session = None
        self.access_token = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self._fetch_token()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _fetch_token(self):
        """Fetch the access token from the authentication endpoint."""
        logger.info(f"Fetching access token from {self.token_endpoint}")
        auth_payload = {"username": self.username, "password": self.password}
        try:
            async with self.session.post(self.token_endpoint, json=auth_payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to get access token: {response.status} - {error_text}")
                    raise Exception("Could not authenticate with RAG service.")

                data = await response.json()
                self.access_token = data.get("access")
                if not self.access_token:
                    raise ValueError("'access' key not found in token response.")
                logger.info("Successfully fetched access token.")
        except Exception as e:
            logger.error(f"Error fetching token: {e}")
            raise

    def _format_legal_prompt(self, question: str) -> str:
        """Format the prompt according to the chat template with specific 5-category legal format"""
        # Create the specific user prompt that includes instructions + question
        user_prompt = f"""Tu es un expert juridique français spécialisé en droit des affaires et droit commercial. 

ÉTAPE 1: Effectue d'abord une analyse complète avec tes tokens de raisonnement.

ÉTAPE 2: Après ton analyse, termine par ces 5 éléments dans cet ordre précis:
• Action Requise: [Action concrète à effectuer] parce que [référence légale précise avec numéro d'article]
• Délai Legal: [Timeframe ou délai applicable] parce que [référence légale précise avec numéro d'article]
• Documents Obligatoires: [Documents nécessaires] parce que [référence légale précise avec numéro d'article]
• Impact Financier: [Coûts, frais ou impact financier] parce que [référence légale précise avec numéro d'article]
• Conséquences Non-Conformité: [Risques en cas de non-respect] parce que [référence légale précise avec numéro d'article]

RÈGLES OBLIGATOIRES:
- Commence chaque ligne par "• [Catégorie]:"
- Termine chaque point par "parce que [justification légale]"
- Cite des articles précis (ex: "article 1193 du Code civil", "article L. 136-1 du Code de la consommation")
- Utilise des détails spécifiques (délais en jours/mois, types de documents, montants)

Question: {question}"""

        # Format exactly like the working curl example with forced reasoning
        prompt_string = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        prompt_string += "<|im_start|>assistant\n<|begin_of_reasoning|>\n"

        return prompt_string
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, question: str, sample_id: str) -> str:
        """Generate response from the RAG pipeline using the auth token."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        if not self.access_token:
            await self._fetch_token()

        headers = {"Authorization": f"Bearer {self.access_token}"}
        prompt = self._format_legal_prompt(question)
        payload = {"question": prompt, "sample_id": sample_id}

        try:
            async with self.session.post(self.endpoint, json=payload, headers=headers) as response:
                if response.status == 403:
                    logger.warning("Access token expired. Fetching a new one.")
                    await self._fetch_token()  # Refresh token
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    async with self.session.post(self.endpoint, json=payload, headers=headers) as retry_response:
                        retry_response.raise_for_status()
                        data = await retry_response.json()
                        return data["response"]

                response.raise_for_status()
                data = await response.json()
                return data["response"]
        except Exception as e:
            logger.error(f"Error generating response from RAG pipeline: {e}")
            raise

    def generate_response_sync(self, question: str, sample_id: str) -> str:
        """Synchronous version of generate_response with full auth flow."""
        self._fetch_token_sync() # Always fetch a fresh token for sync calls
        logger.info("Calling generate_response_sync with question: %s", question)

        headers = {"Authorization": f"Bearer {self.access_token}"}
        prompt = self._format_legal_prompt(question)
        payload = {"question": prompt, "sample_id": sample_id}
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["response"]
        except Exception as e:
            logger.error(f"Sync: Error generating response from RAG pipeline: {e}")
            raise

    def _fetch_token_sync(self):
        """Fetch the access token from the authentication endpoint."""
        logger.info(f"Fetching access token from {self.token_endpoint}")
        auth_payload = {"username": self.username, "password": self.password}
        try:
            response = requests.post(self.token_endpoint, json=auth_payload)
            response.raise_for_status()
            data = response.json()
            self.access_token = data.get("access")
            if not self.access_token:
                raise ValueError("'access' key not found in token response.")
            logger.info("Successfully fetched access token.")
        except Exception as e:
            logger.error(f"Error fetching token: {e}")
            raise
