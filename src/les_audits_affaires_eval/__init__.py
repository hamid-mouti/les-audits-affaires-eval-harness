"""
Les Audits-Affaires LLM Evaluation Harness

Framework d'évaluation pour les modèles de langage sur le benchmark juridique français
'Les Audits-Affaires' de legmlai.

Evaluation framework for Large Language Models on the French legal benchmark
'Les Audits-Affaires' from legmlai.
"""

__version__ = "1.1.0"
__author__ = "LegML Team"
__email__ = "contact@legml.ai"

from .clients import (
    ChatModelClient,
    EvaluatorClient,
    ModelClient,
    StrictChatModelClient,
)
from .evaluation import LesAuditsAffairesEvaluator
from .utils import (
    create_score_distribution_plot,
    export_results_to_excel,
    generate_analysis_report,
    load_evaluation_results,
)

__all__ = [
    "LesAuditsAffairesEvaluator",
    "ModelClient",
    "ChatModelClient",
    "StrictChatModelClient",
    "EvaluatorClient",
    "load_evaluation_results",
    "generate_analysis_report",
    "create_score_distribution_plot",
    "export_results_to_excel",
]
