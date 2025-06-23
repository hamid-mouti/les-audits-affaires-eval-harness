"""
Model client for interfacing with the LLM being evaluated and the Azure OpenAI evaluator
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import *

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


class ModelClient:
    """Client for the model being evaluated"""

    def __init__(self, endpoint: str = MODEL_ENDPOINT, model_name: str = MODEL_NAME):
        self.endpoint = endpoint
        self.model_name = model_name
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _format_prompt(self, question: str) -> str:
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

    def _extract_solution_content(self, response: str) -> str:
        """Extract the content between solution tags directly for the evaluator"""
        try:

            # Debug: Log response length and check for tags
            logger.info(f"Raw response length: {len(response)} characters")
            logger.info(f"Response starts with: {response[:200]}...")
            logger.info(f"Response ends with: ...{response[-200:]}")

            # Check if solution tags exist in response
            start_tag = SOLUTION_START_TAG
            end_tag = SOLUTION_END_TAG
            logger.info(f"Looking for start tag: {start_tag}")
            logger.info(f"Looking for end tag: {end_tag}")
            logger.info(f"Start tag found: {start_tag in response}")
            logger.info(f"End tag found: {end_tag in response}")

            # Primary approach: look for solution tags and extract whatever is between them
            if EXTRACT_SOLUTION_TAGS:
                if start_tag in response:
                    start_idx = response.find(start_tag) + len(start_tag)

                    if end_tag in response:
                        # Both tags found - extract between them
                        end_idx = response.find(end_tag)
                        if start_idx < end_idx:
                            solution_content = response[start_idx:end_idx].strip()
                            logger.info(
                                f"✅ Extracted solution content from both tags: {len(solution_content)} characters"
                            )
                            logger.info(f"Solution content preview: {solution_content[:300]}...")
                            return solution_content
                        else:
                            logger.warning(
                                f"Invalid tag positions: start_idx={start_idx}, end_idx={end_idx}"
                            )
                    else:
                        # Only start tag found - extract from start tag to end of response
                        solution_content = response[start_idx:].strip()
                        logger.info(
                            f"⚠️ Only start tag found, extracting to end: {len(solution_content)} characters"
                        )
                        logger.info(f"Solution content preview: {solution_content[:300]}...")
                        return solution_content
                else:
                    logger.warning(
                        f"Missing solution tags - start: {start_tag in response}, end: {end_tag in response}"
                    )

            # Fallback: return original response if no solution tags found
            logger.warning("❌ No solution tags found, using full response")
            return response

        except Exception as e:
            logger.error(f"Error extracting solution content: {e}")
            return response

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, question: str) -> str:
        """Generate response from the model being evaluated"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        prompt = self._format_prompt(question)

        payload = {
            "prompt": prompt,
            "stream": False,
            "max_new_tokens": MAX_TOKENS,
            "temperature": 0.01,
        }

        try:
            # Increased timeout for longer responses with reasoning tokens (up to 10K tokens)
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for reasoning generation
            async with self.session.post(self.endpoint, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Model API error {response.status}: {error_text}")
                    raise Exception(f"Model API error {response.status}: {error_text}")

                result = await response.json()

                # Handle different response formats
                raw_response = ""
                if isinstance(result, dict):
                    if "generated_text" in result:
                        raw_response = result["generated_text"]
                    elif "text" in result:
                        raw_response = result["text"]
                    elif "content" in result:
                        raw_response = result["content"]
                    elif "message" in result:
                        raw_response = result["message"]
                    elif "response" in result:
                        raw_response = result["response"]
                    else:
                        # If it's a dict but doesn't have expected keys, convert to string
                        raw_response = json.dumps(result)
                elif isinstance(result, str):
                    raw_response = result
                else:
                    raw_response = str(result)

                # Extract solution content if configured
                return self._extract_solution_content(raw_response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_response_sync(self, question: str) -> str:
        """Synchronous version for single requests"""
        prompt = self._format_prompt(question)

        payload = {
            "prompt": prompt,
            "stream": False,
            "max_new_tokens": MAX_TOKENS,
            "temperature": 0.01,
        }

        try:
            # Increased timeout for longer responses with reasoning tokens (up to 10K tokens)
            response = requests.post(self.endpoint, json=payload, timeout=300)  # 5 minutes
            response.raise_for_status()

            result = response.json()

            # Handle different response formats
            raw_response = ""
            if isinstance(result, dict):
                if "generated_text" in result:
                    raw_response = result["generated_text"]
                elif "text" in result:
                    raw_response = result["text"]
                elif "content" in result:
                    raw_response = result["content"]
                elif "message" in result:
                    raw_response = result["message"]
                elif "response" in result:
                    raw_response = result["response"]
                else:
                    raw_response = json.dumps(result)
            elif isinstance(result, str):
                raw_response = result
            else:
                raw_response = str(result)

            # Extract solution content if configured
            return self._extract_solution_content(raw_response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


class EvaluatorClient:
    """Flexible client for LLM evaluation supporting multiple providers"""

    def __init__(self):
        # Check for external evaluator provider first
        self.evaluator_provider = os.getenv("EVALUATOR_PROVIDER", "azure").lower()
        self.evaluator_model = os.getenv("EVALUATOR_MODEL", "gpt-4o")

        logger.info(
            f"Initializing evaluator with provider: {self.evaluator_provider}, model: {self.evaluator_model}"
        )

        if self.evaluator_provider == "azure":
            self._init_azure_openai()
        elif self.evaluator_provider == "openai":
            self._init_openai()
        elif self.evaluator_provider == "mistral":
            self._init_mistral()
        elif self.evaluator_provider == "claude":
            self._init_claude()
        elif self.evaluator_provider == "gemini":
            self._init_gemini()
        elif self.evaluator_provider == "local":
            self._init_local()
        else:
            logger.warning(
                f"Unknown evaluator provider: {self.evaluator_provider}, falling back to Azure OpenAI"
            )
            self._init_azure_openai()

    def _init_azure_openai(self):
        """Initialize Azure OpenAI evaluator"""
        self.client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
        self.client_type = "azure_openai"

    def _init_openai(self):
        """Initialize OpenAI evaluator"""
        from openai import OpenAI

        api_key = os.getenv("EVALUATOR_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "EVALUATOR_OPENAI_API_KEY or OPENAI_API_KEY required for OpenAI evaluator"
            )
        self.client = OpenAI(api_key=api_key)
        self.client_type = "openai"

    def _init_mistral(self):
        """Initialize Mistral evaluator"""
        api_key = os.getenv("EVALUATOR_MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "EVALUATOR_MISTRAL_API_KEY or MISTRAL_API_KEY required for Mistral evaluator"
            )
        self.mistral_api_key = api_key
        self.mistral_endpoint = "https://api.mistral.ai/v1/chat/completions"
        self.client_type = "mistral"

    def _init_claude(self):
        """Initialize Claude evaluator"""
        api_key = os.getenv("EVALUATOR_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "EVALUATOR_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY required for Claude evaluator"
            )
        self.claude_api_key = api_key
        self.client_type = "claude"

    def _init_gemini(self):
        """Initialize Gemini evaluator"""
        api_key = os.getenv("EVALUATOR_GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "EVALUATOR_GOOGLE_API_KEY or GOOGLE_API_KEY required for Gemini evaluator"
            )
        self.gemini_api_key = api_key
        self.client_type = "gemini"

    def _init_local(self):
        """Initialize local model evaluator"""
        self.local_endpoint = os.getenv("EVALUATOR_ENDPOINT") or os.getenv("MODEL_ENDPOINT")
        if not self.local_endpoint:
            raise ValueError("EVALUATOR_ENDPOINT or MODEL_ENDPOINT required for local evaluator")
        self.client_type = "local"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def evaluate_response(
        self, question: str, model_response: str, ground_truth: Dict[str, str]
    ) -> Dict[str, Any]:
        """Evaluate a model response using the configured evaluator"""

        # Format the evaluation prompt
        evaluation_prompt = LLM_EVALUATION_PROMPT.format(
            user_question=question,
            model_response=model_response,
            action_requise=ground_truth.get("action_requise", ""),
            delai_legal=ground_truth.get("delai_legal", ""),
            documents_obligatoires=ground_truth.get("documents_obligatoires", ""),
            impact_financier=ground_truth.get("impact_financier", ""),
            consequences_non_conformite=ground_truth.get("consequences_non_conformite", ""),
        )

        try:
            if self.client_type == "azure_openai":
                return self._evaluate_azure_openai(evaluation_prompt)
            elif self.client_type == "openai":
                return self._evaluate_openai(evaluation_prompt)
            elif self.client_type == "mistral":
                return self._evaluate_mistral(evaluation_prompt)
            elif self.client_type == "claude":
                return self._evaluate_claude(evaluation_prompt)
            elif self.client_type == "gemini":
                return self._evaluate_gemini(evaluation_prompt)
            elif self.client_type == "local":
                return self._evaluate_local(evaluation_prompt)
            else:
                logger.error(f"Unknown client type: {self.client_type}")
                return self._create_default_evaluation()

        except Exception as e:
            logger.error(f"Error during evaluation with {self.client_type}: {e}")
            return self._create_default_evaluation()

    def _evaluate_azure_openai(self, evaluation_prompt: str) -> Dict[str, Any]:
        """Evaluate using Azure OpenAI"""
        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.1,
            max_tokens=12000,
            response_format={"type": "json_object"},
        )
        return self._parse_evaluation_response(response.choices[0].message.content)

    def _evaluate_openai(self, evaluation_prompt: str) -> Dict[str, Any]:
        """Evaluate using OpenAI"""
        response = self.client.chat.completions.create(
            model=self.evaluator_model,
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.1,
            max_tokens=12000,
            response_format={"type": "json_object"},
        )
        return self._parse_evaluation_response(response.choices[0].message.content)

    def _evaluate_mistral(self, evaluation_prompt: str) -> Dict[str, Any]:
        """Evaluate using Mistral"""

        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.evaluator_model,
            "messages": [{"role": "user", "content": evaluation_prompt}],
            "temperature": 0.1,
            "max_tokens": 12000,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(self.mistral_endpoint, json=payload, headers=headers, timeout=300)
        response.raise_for_status()

        result = response.json()
        return self._parse_evaluation_response(result["choices"][0]["message"]["content"])

    def _evaluate_claude(self, evaluation_prompt: str) -> Dict[str, Any]:
        """Evaluate using Claude"""

        headers = {
            "x-api-key": self.claude_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Add JSON format instruction to prompt for Claude
        claude_prompt = (
            evaluation_prompt
            + "\n\nRéponds UNIQUEMENT avec un objet JSON valide, sans texte supplémentaire."
        )

        payload = {
            "model": self.evaluator_model,
            "max_tokens": 12000,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": claude_prompt}],
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=300
        )
        response.raise_for_status()

        result = response.json()
        return self._parse_evaluation_response(result["content"][0]["text"])

    def _evaluate_gemini(self, evaluation_prompt: str) -> Dict[str, Any]:
        """Evaluate using Gemini"""

        # Add JSON format instruction to prompt for Gemini
        gemini_prompt = (
            evaluation_prompt
            + "\n\nRéponds UNIQUEMENT avec un objet JSON valide, sans texte supplémentaire."
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.evaluator_model}:generateContent?key={self.gemini_api_key}"

        payload = {
            "contents": [{"parts": [{"text": gemini_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 12000,
                "responseMimeType": "application/json",
            },
        }

        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        return self._parse_evaluation_response(
            result["candidates"][0]["content"]["parts"][0]["text"]
        )

    def _evaluate_local(self, evaluation_prompt: str) -> Dict[str, Any]:
        """Evaluate using local model"""

        # Add JSON format instruction to prompt for local models
        local_prompt = (
            evaluation_prompt
            + "\n\nRéponds UNIQUEMENT avec un objet JSON valide, sans texte supplémentaire."
        )

        # Try chat endpoint first, then generate endpoint
        for endpoint_suffix in ["/chat", "/generate"]:
            try:
                endpoint = self.local_endpoint.rstrip("/") + endpoint_suffix

                if endpoint_suffix == "/chat":
                    payload = {
                        "messages": [{"role": "user", "content": local_prompt}],
                        "temperature": 0.1,
                        "max_tokens": 12000,
                        "stream": False,
                    }
                else:
                    payload = {
                        "prompt": local_prompt,
                        "temperature": 0.1,
                        "max_new_tokens": 12000,
                        "stream": False,
                    }

                response = requests.post(endpoint, json=payload, timeout=300)
                response.raise_for_status()

                result = response.json()

                # Extract response text from various formats
                response_text = ""
                if isinstance(result, dict):
                    if "choices" in result and result["choices"]:
                        if "message" in result["choices"][0]:
                            response_text = result["choices"][0]["message"]["content"]
                        elif "text" in result["choices"][0]:
                            response_text = result["choices"][0]["text"]
                    elif "generated_text" in result:
                        response_text = result["generated_text"]
                    elif "text" in result:
                        response_text = result["text"]
                    elif "content" in result:
                        response_text = result["content"]
                    else:
                        response_text = json.dumps(result)
                elif isinstance(result, str):
                    response_text = result
                else:
                    response_text = str(result)

                return self._parse_evaluation_response(response_text)

            except Exception as e:
                logger.warning(f"Failed to evaluate with {endpoint}: {e}")
                continue

        raise Exception(f"Failed to evaluate with local model at {self.local_endpoint}")

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate evaluation response"""
        try:
            # Clean up response text
            response_text = response_text.strip()

            # Try to extract JSON if it's wrapped in other text
            if not response_text.startswith("{"):
                # Look for JSON object in the text
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group()

            result = json.loads(response_text)

            # Validate and fix the response structure
            required_scores = [
                "action_requise",
                "delai_legal",
                "documents_obligatoires",
                "impact_financier",
                "consequences_non_conformite",
            ]

            # Ensure nested dicts exist
            result.setdefault("scores", {})
            result.setdefault("justifications", {})

            for key in required_scores:
                result["scores"].setdefault(key, 0)
                result["justifications"].setdefault(key, "N/A")

            # Calculate global score if missing
            if "score_global" not in result or not isinstance(result["score_global"], (int, float)):
                scores_vals = list(result["scores"].values())
                if scores_vals:
                    result["score_global"] = sum(scores_vals) / len(scores_vals)
                else:
                    result["score_global"] = 0

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {response_text}")
            logger.error(f"JSON decode error: {e}")
            return self._create_default_evaluation()

    def _create_default_evaluation(self) -> Dict[str, Any]:
        """Create a default evaluation response when evaluation fails"""
        return {
            "score_global": 0,
            "scores": {
                "action_requise": 0,
                "delai_legal": 0,
                "documents_obligatoires": 0,
                "impact_financier": 0,
                "consequences_non_conformite": 0,
            },
            "justifications": {
                "action_requise": "Évaluation échouée",
                "delai_legal": "Évaluation échouée",
                "documents_obligatoires": "Évaluation échouée",
                "impact_financier": "Évaluation échouée",
                "consequences_non_conformite": "Évaluation échouée",
            },
        }


class ChatModelClient:
    """Client for the model being evaluated using chat endpoint"""

    def __init__(self, endpoint: str = MODEL_ENDPOINT, model_name: str = MODEL_NAME):
        # Convert generate endpoint to chat endpoint
        if endpoint.endswith("/generate"):
            self.endpoint = endpoint.replace("/generate", "/chat")
        else:
            self.endpoint = endpoint + "/chat" if not endpoint.endswith("/chat") else endpoint
        self.model_name = model_name
        self.session = None
        # --- Mistral support ---
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.is_mistral = "mistral.ai" in self.endpoint
        self.headers = (
            {"Authorization": f"Bearer {self.api_key}"} if self.is_mistral and self.api_key else {}
        )
        # Normalise Mistral endpoint (avoid duplicate '/chat')
        if self.is_mistral:
            base_url = self.endpoint.rstrip("/")
            # Remove a trailing '/chat' if present (to avoid duplication)
            if base_url.endswith("/chat"):
                base_url = base_url[:-5]  # strip '/chat'
            # Ensure single '/chat/completions'
            self.endpoint = base_url + "/chat/completions"

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _format_chat_messages(self, question: str) -> List[Dict[str, str]]:
        """Format the question as simple chat messages without special formatting"""
        return [{"role": "user", "content": question}]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, question: str) -> str:
        """Generate response from the model being evaluated using chat endpoint"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        messages = self._format_chat_messages(question)

        if self.is_mistral:
            payload = {
                "model": os.getenv("MISTRAL_MODEL_ID", self.model_name),
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "stream": False,
            }
        else:
            payload = {
                "messages": messages,
                "stream": False,
                "max_new_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": 0.9,
                "do_sample": True,
            }

        try:
            # Increased timeout for longer responses
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
            async with self.session.post(self.endpoint, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Model API error {response.status}: {error_text}")
                    raise Exception(f"Model API error {response.status}: {error_text}")

                result = await response.json()

                # Handle different response formats
                raw_response = ""
                if isinstance(result, dict):
                    if "generated_text" in result:
                        raw_response = result["generated_text"]
                    elif "text" in result:
                        raw_response = result["text"]
                    elif "content" in result:
                        raw_response = result["content"]
                    elif "message" in result:
                        raw_response = result["message"]
                    elif "response" in result:
                        raw_response = result["response"]
                    else:
                        # If it's a dict but doesn't have expected keys, convert to string
                        raw_response = json.dumps(result)
                elif isinstance(result, str):
                    raw_response = result
                else:
                    raw_response = str(result)

                # Return response without solution tag extraction
                return raw_response.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_response_sync(self, question: str) -> str:
        """Synchronous version for single requests"""
        messages = self._format_chat_messages(question)

        if self.is_mistral:
            payload = {
                "model": os.getenv("MISTRAL_MODEL_ID", self.model_name),
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "stream": False,
            }
        else:
            payload = {
                "messages": messages,
                "stream": False,
                "max_new_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": 0.9,
                "do_sample": True,
            }

        try:
            # Increased timeout for longer responses
            response = requests.post(
                self.endpoint, json=payload, headers=self.headers, timeout=300
            )  # 5 minutes
            response.raise_for_status()

            result = response.json()

            # Handle different response formats
            raw_response = ""
            if isinstance(result, dict):
                if "generated_text" in result:
                    raw_response = result["generated_text"]
                elif "text" in result:
                    raw_response = result["text"]
                elif "content" in result:
                    raw_response = result["content"]
                elif "message" in result:
                    raw_response = result["message"]
                elif "response" in result:
                    raw_response = result["response"]
                else:
                    raw_response = json.dumps(result)
            elif isinstance(result, str):
                raw_response = result
            else:
                raw_response = str(result)

            # Return response without solution tag extraction
            return raw_response.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


class StrictChatModelClient:
    """Client for the model being evaluated using chat endpoint with strict formatting and repetition handling"""

    def __init__(self, endpoint: str = MODEL_ENDPOINT, model_name: str = MODEL_NAME):
        # Convert generate endpoint to chat endpoint
        if endpoint.endswith("/generate"):
            self.endpoint = endpoint.replace("/generate", "/chat")
        else:
            self.endpoint = endpoint + "/chat" if not endpoint.endswith("/chat") else endpoint
        self.model_name = model_name
        self.session = None
        # --- Mistral support ---
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.is_mistral = "mistral.ai" in self.endpoint
        self.headers = (
            {"Authorization": f"Bearer {self.api_key}"} if self.is_mistral and self.api_key else {}
        )
        # Normalise Mistral endpoint (avoid duplicate '/chat')
        if self.is_mistral:
            base_url = self.endpoint.rstrip("/")
            # Remove a trailing '/chat' if present (to avoid duplication)
            if base_url.endswith("/chat"):
                base_url = base_url[:-5]  # strip '/chat'
            # Ensure single '/chat/completions'
            self.endpoint = base_url + "/chat/completions"

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _format_strict_chat_messages(self, question: str) -> List[Dict[str, str]]:
        """Format the question with encouraging instructions for proper formatting"""
        encouraging_prompt = f"""Tu es un expert juridique français spécialisé en droit des affaires et droit commercial.

Réponds à la question juridique ci-dessous, puis termine ta réponse par un résumé structuré avec ces 5 éléments:

• Action Requise: [décris l'action concrète nécessaire] parce que [référence légale précise]
• Délai Legal: [indique le délai précis] parce que [référence légale précise]  
• Documents Obligatoires: [liste les documents nécessaires] parce que [référence légale précise]
• Impact Financier: [estime les coûts/frais] parce que [référence légale précise]
• Conséquences Non-Conformité: [explique les risques] parce que [référence légale précise]

Conseil: Utilise des références légales différentes pour chaque catégorie et sois précis dans tes explications.

Question: {question}"""

        return [{"role": "user", "content": encouraging_prompt}]

    def _detect_repetition(self, text: str) -> bool:
        """Detect if the response has excessive repetition"""
        if not text or len(text) < 100:
            return False

        # Split into sentences and check for repetition
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        if len(sentences) < 3:
            return False

        # Check for repeated sentences
        unique_sentences = set(sentences)
        repetition_ratio = 1 - (len(unique_sentences) / len(sentences))

        # Check for repeated phrases
        words = text.lower().split()
        if len(words) < 20:
            return False

        # Look for repeated 3-word phrases
        phrases = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        unique_phrases = set(phrases)
        phrase_repetition = 1 - (len(unique_phrases) / len(phrases))

        # Detection thresholds
        is_repetitive = repetition_ratio > 0.3 or phrase_repetition > 0.4

        if is_repetitive:
            logger.warning(
                f"Repetition detected - sentences: {repetition_ratio:.2f}, phrases: {phrase_repetition:.2f}"
            )

        return is_repetitive

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, question: str) -> str:
        """Generate response with repetition detection and retry logic"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        messages = self._format_strict_chat_messages(question)

        # Try with different parameters if repetition is detected
        for attempt in range(3):
            # Adjust parameters based on attempt
            if attempt == 0:
                # First attempt - EXTREMELY LOW TEMP parameters (T=0.01 OPTIMAL)
                # BREAKTHROUGH results: 76% avg format compliance, 80% excellent responses, 0% repetition
                temperature = 0.01  # Extremely low temperature for maximum determinism
                repetition_penalty = 1.0  # Minimal repetition penalty to avoid conflicts
                top_p = 0.9  # Standard top_p for deterministic output
            elif attempt == 1:
                # Second attempt - Ultra-Low-Temp fallback
                temperature = 0.1
                repetition_penalty = 1.0
                top_p = 0.9
            else:
                # Third attempt - Low-Temp fallback
                temperature = 0.2
                repetition_penalty = 1.05
                top_p = 0.95

            if self.is_mistral:
                payload = {
                    "model": os.getenv("MISTRAL_MODEL_ID", self.model_name),
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "temperature": temperature,
                    "stream": False,
                }
            else:
                payload = {
                    "messages": messages,
                    "stream": False,
                    "max_new_tokens": MAX_TOKENS,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                    "repetition_penalty": repetition_penalty,
                }

            try:
                timeout = aiohttp.ClientTimeout(total=300)
                async with self.session.post(
                    self.endpoint, json=payload, timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Model API error {response.status}: {error_text}")
                        raise Exception(f"Model API error {response.status}: {error_text}")

                    result = await response.json()

                    # Extract response text
                    raw_response = ""
                    if isinstance(result, dict):
                        if "generated_text" in result:
                            raw_response = result["generated_text"]
                        elif "text" in result:
                            raw_response = result["text"]
                        elif "content" in result:
                            raw_response = result["content"]
                        elif "message" in result:
                            raw_response = result["message"]
                        elif "response" in result:
                            raw_response = result["response"]
                        else:
                            raw_response = json.dumps(result)
                    elif isinstance(result, str):
                        raw_response = result
                    else:
                        raw_response = str(result)

                    response_text = raw_response.strip()

                    # Check for repetition
                    if self._detect_repetition(response_text):
                        if attempt < 2:  # Not the last attempt
                            logger.warning(
                                f"Repetition detected on attempt {attempt + 1}, retrying with adjusted parameters"
                            )
                            continue
                        else:
                            logger.warning(
                                f"Repetition still detected on final attempt, returning response anyway"
                            )

                    # Check format compliance with flexible patterns
                    section_patterns = {
                        "Action Requise": [
                            "Action Requise:",
                            "Action requise:",
                            "• Action Requise",
                            "**Action Requise",
                        ],
                        "Délai Legal": [
                            "Délai Legal:",
                            "Délais Legal:",
                            "Délai légal:",
                            "Délais légal:",
                            "• Délai Legal",
                            "**Délai Legal",
                            "**Délais Legal",
                        ],
                        "Documents Obligatoires": [
                            "Documents Obligatoires:",
                            "Documents obligatoires:",
                            "Documents Emploi:",
                            "• Documents Obligatoires",
                            "**Documents Obligatoires",
                            "**Documents Emploi",
                        ],
                        "Impact Financier": [
                            "Impact Financier:",
                            "Impact financier:",
                            "• Impact Financier",
                            "**Impact Financier",
                        ],
                        "Conséquences Non-Conformité": [
                            "Conséquences Non-Conformité:",
                            "Conséquences non-conformité:",
                            "Conséquences Non-conséquence:",
                            "• Conséquences Non-Conformité",
                            "**Conséquences Non-Conformité",
                        ],
                    }

                    found_sections = []
                    missing_sections = []

                    for section, patterns in section_patterns.items():
                        found = any(pattern in response_text for pattern in patterns)
                        if found:
                            found_sections.append(section)
                        else:
                            missing_sections.append(section)

                    if missing_sections:
                        logger.warning(f"Missing format sections: {missing_sections}")
                    if found_sections:
                        logger.info(f"Found format sections: {found_sections}")

                    return response_text

            except Exception as e:
                if attempt == 2:  # Last attempt
                    logger.error(f"Error generating response on final attempt: {e}")
                    raise
                else:
                    logger.warning(f"Error on attempt {attempt + 1}: {e}, retrying...")
                    continue

    def generate_response_sync(self, question: str) -> str:
        """Synchronous version with repetition handling"""
        messages = self._format_strict_chat_messages(question)

        for attempt in range(3):
            # Adjust parameters based on attempt
            if attempt == 0:
                # First attempt - EXTREMELY LOW TEMP parameters (T=0.01 OPTIMAL)
                # BREAKTHROUGH results: 76% avg format compliance, 80% excellent responses, 0% repetition
                temperature = 0.01  # Extremely low temperature for maximum determinism
                repetition_penalty = 1.0  # Minimal repetition penalty to avoid conflicts
                top_p = 0.9  # Standard top_p for deterministic output
            elif attempt == 1:
                # Second attempt - Ultra-Low-Temp fallback
                temperature = 0.1
                repetition_penalty = 1.0
                top_p = 0.9
            else:
                # Third attempt - Low-Temp fallback
                temperature = 0.2
                repetition_penalty = 1.05
                top_p = 0.95

            if self.is_mistral:
                payload = {
                    "model": os.getenv("MISTRAL_MODEL_ID", self.model_name),
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "temperature": temperature,
                    "stream": False,
                }
            else:
                payload = {
                    "messages": messages,
                    "stream": False,
                    "max_new_tokens": MAX_TOKENS,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                    "repetition_penalty": repetition_penalty,
                }

            try:
                response = requests.post(
                    self.endpoint, json=payload, headers=self.headers, timeout=300
                )
                response.raise_for_status()

                result = response.json()

                # Extract response text
                raw_response = ""
                if isinstance(result, dict):
                    if "generated_text" in result:
                        raw_response = result["generated_text"]
                    elif "text" in result:
                        raw_response = result["text"]
                    elif "content" in result:
                        raw_response = result["content"]
                    elif "message" in result:
                        raw_response = result["message"]
                    elif "response" in result:
                        raw_response = result["response"]
                    else:
                        raw_response = json.dumps(result)
                elif isinstance(result, str):
                    raw_response = result
                else:
                    raw_response = str(result)

                response_text = raw_response.strip()

                # Check for repetition
                if self._detect_repetition(response_text):
                    if attempt < 2:
                        logger.warning(
                            f"Repetition detected on attempt {attempt + 1}, retrying with adjusted parameters"
                        )
                        continue
                    else:
                        logger.warning(
                            f"Repetition still detected on final attempt, returning response anyway"
                        )

                return response_text

            except Exception as e:
                if attempt == 2:
                    logger.error(f"Error generating response on final attempt: {e}")
                    raise
                else:
                    logger.warning(f"Error on attempt {attempt + 1}: {e}, retrying...")
                    continue
