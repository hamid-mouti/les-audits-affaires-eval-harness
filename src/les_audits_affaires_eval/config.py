"""
Configuration file for Les Audits-Affaires LLM Evaluation Harness
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Evaluator Configuration (Flexible LLM Support)
EVALUATOR_PROVIDER = os.getenv(
    "EVALUATOR_PROVIDER", "azure"
).lower()  # azure, openai, mistral, claude, gemini, local
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-4o")

# Evaluator API Keys (fallback to main keys if evaluator-specific keys not set)
EVALUATOR_OPENAI_API_KEY = os.getenv("EVALUATOR_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
EVALUATOR_MISTRAL_API_KEY = os.getenv("EVALUATOR_MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
EVALUATOR_ANTHROPIC_API_KEY = os.getenv("EVALUATOR_ANTHROPIC_API_KEY") or os.getenv(
    "ANTHROPIC_API_KEY"
)
EVALUATOR_GOOGLE_API_KEY = os.getenv("EVALUATOR_GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
EVALUATOR_ENDPOINT = os.getenv("EVALUATOR_ENDPOINT")  # For local evaluator models

# Model to Evaluate Configuration
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME", "legml-d_affairs-14b")

# Evaluation Configuration
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))  # Increased for high-throughput processing
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(
    os.getenv("MAX_TOKENS", "32768")
)  # Increased to 32K tokens to ensure completion of solution tags
CONCURRENT_REQUESTS = int(
    os.getenv("CONCURRENT_REQUESTS", "50")
)  # Support up to 200 concurrent requests

# Dataset Configuration
DATASET_NAME = "legmlai/les-audits-affaires"
DATASET_SPLIT = "train"

# System Prompt for the Model Being Evaluated
SYSTEM_PROMPT = """Vous êtes Legml d'affaires, assistant IA expert en finance, économie et business. 
Vous fournissez des réponses fiables, structurées et orientées action pour aider les professionnels.

IMPORTANT: Après votre analyse complète avec les tokens de raisonnement, vous DEVEZ terminer par une section solution délimitée par les balises <|begin_of_solution|> et <|end_of_solution|> contenant uniquement la réponse finale structurée.

Format requis:
<|begin_of_solution|>
**Action requise** : [actions légales nécessaires]
**Délai légal** : [échéances et délais]
**Documents obligatoires** : [documentation requise]
**Impact financier** : [implications financières]
**Conséquences non-conformité** : [conséquences de la non-conformité]
<|end_of_solution|>"""

# Evaluation Prompt Template
LLM_EVALUATION_PROMPT = """
Tu es un juriste-expert français, spécialiste du Code civil et du droit des affaires.  
Ta mission : évaluer la qualité d'une réponse LLM dans le cadre du benchmark « Les Audits-Affaires ».

Barème d'évaluation :  
- 5 rubriques, chacune notée sur 100 pts :  
  1. action_requise  
  2. delai_legal  
  3. documents_obligatoires  
  4. impact_financier  
  5. consequences_non_conformite  
- Le score global est la moyenne arithmétique simple des 5 rubriques.  
- Pour chaque rubrique, évalue selon : exactitude juridique, concordance avec le ground truth, clarté et justification (article cité + explication).

---

"question": "{user_question}",

"model_response": "{model_response}",

"ground_truth": {{
  "action_requise": "{action_requise}",
  "delai_legal": "{delai_legal}",
  "documents_obligatoires": "{documents_obligatoires}",
  "impact_financier": "{impact_financier}",
  "consequences_non_conformite": "{consequences_non_conformite}"
}}

---

Donne ta réponse STRICTEMENT dans ce format JSON :  

{{
  "score_global": 0,
  "scores": {{
    "action_requise": 0,
    "delai_legal": 0,
    "documents_obligatoires": 0,
    "impact_financier": 0,
    "consequences_non_conformite": 0
  }},
  "justifications": {{
    "action_requise": "",
    "delai_legal": "",
    "documents_obligatoires": "",
    "impact_financier": "",
    "consequences_non_conformite": ""
  }}
}}
"""


# Dynamic Results Directory Configuration
def get_safe_model_name(model_name: str) -> str:
    """Convert model name to safe directory name"""
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_").replace(":", "_")


# Output Configuration - Dynamic based on model name
BASE_RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
MODEL_SAFE_NAME = get_safe_model_name(MODEL_NAME)

# If a custom RESULTS_DIR is set (like from environment), use it directly
# Otherwise, create model-specific subdirectory automatically
if os.getenv("RESULTS_DIR") and os.getenv("RESULTS_DIR") != "results":
    # Custom results dir is set, use it as-is
    RESULTS_DIR = os.getenv("RESULTS_DIR")
else:
    # Default behavior: create model-specific subdirectory
    RESULTS_DIR = f"{BASE_RESULTS_DIR}/{MODEL_SAFE_NAME}"

OUTPUT_FILE = "evaluation_results.json"
SUMMARY_FILE = "evaluation_summary.json"
DETAILED_FILE = "detailed_results.jsonl"

# Solution Extraction Configuration
EXTRACT_SOLUTION_TAGS = os.getenv("EXTRACT_SOLUTION_TAGS", "true").lower() in (
    "true",
    "1",
    "yes",
    "y",
)
SOLUTION_START_TAG = "<|begin_of_solution|>"
SOLUTION_END_TAG = "<|end_of_solution|>"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "evaluation.log"
