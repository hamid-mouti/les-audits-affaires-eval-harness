
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, question: str) -> str:
        """Generate response from the RAG pipeline using the auth token."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        if not self.access_token:
            raise RuntimeError("Access token not available. Authentication might have failed.")

        headers = {"Authorization": f"Bearer {self.access_token}"}
        payload = {"question": question}

        try:
            async with self.session.post(self.endpoint, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"RAG API error {response.status}: {error_text}")
                    raise Exception(f"RAG API error {response.status}: {error_text}")

                result = await response.json()
                return result.get("response", "")
        except Exception as e:
            logger.error(f"Error generating response from RAG pipeline: {e}")
            raise

    def generate_response_sync(self, question: str) -> str:
        """Synchronous version of generate_response with full auth flow."""
        # 1. Get token
        auth_payload = {"username": self.username, "password": self.password}
        try:
            response = requests.post(self.token_endpoint, json=auth_payload)
            response.raise_for_status()
            token = response.json().get("access")
            if not token:
                raise ValueError("'access' key not found in token response.")
        except Exception as e:
            logger.error(f"Sync: Failed to get access token: {e}")
            raise

        # 2. Make request
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"question": question}
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Sync: Error generating response from RAG pipeline: {e}")
            raise
