"""
Les Audits-Affaires LLM Evaluation Harness

Framework d'évaluation pour les modèles de langage sur le benchmark juridique français
'Les Audits-Affaires' de legmlai.

Evaluation framework for Large Language Models on the French legal benchmark
'Les Audits-Affaires' from legmlai.
"""

__version__ = "1.1.0"
# __version__ will be set dynamically below
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

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
except ImportError:  # pragma: no cover – Python <3.8, unlikely
    def _pkg_version(_: str) -> str:
        return "0.0.0"
    class PackageNotFoundError(Exception):
        pass

try:
    __version__ = _pkg_version("les-audits-affaires-eval-harness")
except PackageNotFoundError:
    __version__ = "0.0.0"
