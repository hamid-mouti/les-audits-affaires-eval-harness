"""
Utility functions for Les Audits-Affaires evaluation analysis
"""

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import RESULTS_DIR


def load_evaluation_results(results_file: Optional[str] = None) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    if results_file is None:
        results_file = os.path.join(RESULTS_DIR, "evaluation_results.json")

    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_score_distribution_plot(
    results: Dict[str, Any], save_path: Optional[str] = None
) -> None:
    """Create distribution plots for scores"""
    detailed_results = results["detailed_results"]

    # Extract scores
    global_scores = [r["evaluation"]["score_global"] for r in detailed_results]
    category_scores = {
        "action_requise": [r["evaluation"]["scores"]["action_requise"] for r in detailed_results],
        "delai_legal": [r["evaluation"]["scores"]["delai_legal"] for r in detailed_results],
        "documents_obligatoires": [
            r["evaluation"]["scores"]["documents_obligatoires"] for r in detailed_results
        ],
        "impact_financier": [
            r["evaluation"]["scores"]["impact_financier"] for r in detailed_results
        ],
        "consequences_non_conformite": [
            r["evaluation"]["scores"]["consequences_non_conformite"] for r in detailed_results
        ],
    }

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Score Distributions - Les Audits-Affaires Evaluation", fontsize=16)

    # Global score distribution
    axes[0, 0].hist(global_scores, bins=20, alpha=0.7, color="blue")
    axes[0, 0].set_title("Global Score Distribution")
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].set_ylabel("Frequency")

    # Category score distributions
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    colors = ["red", "green", "orange", "purple", "brown"]

    for i, (category, scores) in enumerate(category_scores.items()):
        row, col = positions[i]
        axes[row, col].hist(scores, bins=20, alpha=0.7, color=colors[i])
        axes[row, col].set_title(f'{category.replace("_", " ").title()}')
        axes[row, col].set_xlabel("Score")
        axes[row, col].set_ylabel("Frequency")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.savefig(
            os.path.join(RESULTS_DIR, "score_distributions.png"), dpi=300, bbox_inches="tight"
        )

    plt.show()


def create_correlation_heatmap(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Create correlation heatmap between different score categories"""
    detailed_results = results["detailed_results"]

    # Create DataFrame
    data = []
    for result in detailed_results:
        scores = result["evaluation"]["scores"]
        data.append(
            {
                "global_score": result["evaluation"]["score_global"],
                "action_requise": scores["action_requise"],
                "delai_legal": scores["delai_legal"],
                "documents_obligatoires": scores["documents_obligatoires"],
                "impact_financier": scores["impact_financier"],
                "consequences_non_conformite": scores["consequences_non_conformite"],
            }
        )

    df = pd.DataFrame(data)

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", center=0, square=True, linewidths=0.5
    )
    plt.title("Score Correlation Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.savefig(
            os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=300, bbox_inches="tight"
        )

    plt.show()


def analyze_performance_by_category(results: Dict[str, Any]) -> pd.DataFrame:
    """Analyze performance statistics by category"""
    summary = results["summary"]

    data = []
    for category, stats in summary["category_scores"].items():
        data.append(
            {
                "Category": category.replace("_", " ").title(),
                "Mean Score": stats["mean"],
                "Std Dev": stats["std"],
                "Min Score": stats["min"],
                "Max Score": stats["max"],
            }
        )

    # Add global score
    global_stats = summary["global_score"]
    data.append(
        {
            "Category": "Global Score",
            "Mean Score": global_stats["mean"],
            "Std Dev": global_stats["std"],
            "Min Score": global_stats["min"],
            "Max Score": global_stats["max"],
        }
    )

    df = pd.DataFrame(data)
    return df


def find_challenging_samples(results: Dict[str, Any], n_samples: int = 10) -> List[Dict[str, Any]]:
    """Find the most challenging samples (lowest scores)"""
    detailed_results = results["detailed_results"]

    # Sort by global score
    sorted_results = sorted(detailed_results, key=lambda x: x["evaluation"]["score_global"])

    return sorted_results[:n_samples]


def find_best_samples(results: Dict[str, Any], n_samples: int = 10) -> List[Dict[str, Any]]:
    """Find the best performing samples (highest scores)"""
    detailed_results = results["detailed_results"]

    # Sort by global score (descending)
    sorted_results = sorted(
        detailed_results, key=lambda x: x["evaluation"]["score_global"], reverse=True
    )

    return sorted_results[:n_samples]


def analyze_rag_performance(
    rag_results: Dict[str, Any], all_other_results: List[Dict[str, Any]]
) -> str:
    """
    Analyzes the performance of a RAG model compared to other models.
    Finds the top 20 samples where the RAG model has a high score but is not the top performer.
    """
    rag_model_name = rag_results["summary"]["model_name"]
    report_section = f"### Analysis for RAG Model: {rag_model_name}\n\n"

    # Create a dictionary to store scores and justifications from all models for each sample
    scores = {}

    # Process RAG model results
    for result in rag_results["detailed_results"]:
        sample_idx = result["sample_idx"]
        if "evaluation" in result and "score_global" in result["evaluation"]:
            scores[sample_idx] = {
                "rag_score": result["evaluation"]["score_global"],
                "rag_justifications": result["evaluation"].get("justifications"),
                "question": result["question"],
                "other_scores": {},
                "other_justifications": {},
            }

    # Process other models' results
    for other_results in all_other_results:
        other_model_name = other_results["summary"]["model_name"]
        for result in other_results["detailed_results"]:
            sample_idx = result["sample_idx"]
            if (
                sample_idx in scores
                and "evaluation" in result
                and "score_global" in result["evaluation"]
            ):
                scores[sample_idx]["other_scores"][other_model_name] = result[
                    "evaluation"
                ]["score_global"]
                scores[sample_idx]["other_justifications"][other_model_name] = result[
                    "evaluation"
                ].get("justifications")

    # Filter and analyze
    analysis_results = []
    for sample_idx, data in scores.items():
        if "rag_score" in data and data["other_scores"]:
            max_other_score = max(data["other_scores"].values())
            if data["rag_score"] < max_other_score:
                best_other_model = max(data["other_scores"], key=data["other_scores"].get)
                analysis_results.append(
                    {
                        "sample_idx": sample_idx,
                        "question": data["question"],
                        "rag_score": data["rag_score"],
                        "rag_justifications": data.get("rag_justifications"),
                        "highest_score": max_other_score,
                        "highest_score_model": best_other_model,
                        "highest_score_justifications": data.get(
                            "other_justifications", {}
                        ).get(best_other_model),
                    }
                )

    # Sort by RAG score (descending) and get top 20
    sorted_results = sorted(analysis_results, key=lambda x: x["rag_score"], reverse=True)[:20]

    if not sorted_results:
        report_section += (
            "No samples found where the RAG model was outperformed by another model.\n"
        )
        return report_section

    report_section += "Top 20 samples where the RAG model performed well but was not the top performer:\n\n"
    for result in sorted_results:
        report_section += f"**Sample Index:** {result['sample_idx']}\n\n"
        report_section += f"**Question:** {result['question']}\n\n"
        report_section += f"**{rag_model_name} Score:** {result['rag_score']}\n\n"
        if result["rag_justifications"]:
            report_section += f"**Justifications ({rag_model_name}):**\n\n"
            for category, justification in result["rag_justifications"].items():
                report_section += (
                    f"- **{category.replace('_', ' ').title()}:** {justification}\n\n"
                )
        report_section += f"**Highest Score:** {result['highest_score']} (Model: {result['highest_score_model']})\n\n"
        if result["highest_score_justifications"]:
            report_section += (
                f"**Justifications ({result['highest_score_model']}):**\n\n"
            )
            for category, justification in result[
                "highest_score_justifications"
            ].items():
                report_section += (
                    f"- **{category.replace('_', ' ').title()}:** {justification}\n\n"
                )
        report_section += "---\n\n"

    return report_section


def generate_analysis_report(
    current_results: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    output_file: Optional[str] = None,
) -> str:
    """Generate a comprehensive analysis report"""
    if output_file is None:
        output_file = os.path.join(RESULTS_DIR, "analysis_report.md")

    summary = current_results["summary"]
    model_name = summary["model_name"]

    report = f"""# Les Audits-Affaires Evaluation Report

## Model Information
- **Model Name**: {model_name}
- **Dataset**: {summary['dataset_name']}
- **Evaluation Date**: {summary['evaluation_timestamp']}
- **Total Samples**: {summary['sample_count']}

## Overall Performance

### Global Score
- **Mean**: {summary['global_score']['mean']:.2f}
- **Standard Deviation**: {summary['global_score']['std']:.2f}
- **Range**: {summary['global_score']['min']:.2f} - {summary['global_score']['max']:.2f}

### Success Rate
- **Successful Evaluations**: {summary['successful_evaluations']} ({summary['successful_evaluations']/summary['sample_count']*100:.1f}%)
- **Failed Evaluations**: {summary['failed_evaluations']} ({summary['failed_evaluations']/summary['sample_count']*100:.1f}%)

## Category Performance

"""

    for category, stats in summary["category_scores"].items():
        report += f"### {category.replace('_', ' ').title()}\n"
        report += f"- **Mean**: {stats['mean']:.2f}\n"
        report += f"- **Standard Deviation**: {stats['std']:.2f}\n"
        report += f"- **Range**: {stats['min']:.2f} - {stats['max']:.2f}\n\n"

    report += f"""## Configuration
- **Max Tokens**: {summary['configuration']['max_tokens']}
- **Temperature**: {summary['configuration']['temperature']}
- **Batch Size**: {summary['configuration']['batch_size']}
- **Concurrent Requests**: {summary['configuration']['concurrent_requests']}

## Analysis

### Performance Insights
"""

    # Add some basic insights
    best_category = max(summary["category_scores"].items(), key=lambda x: x[1]["mean"])
    worst_category = min(summary["category_scores"].items(), key=lambda x: x[1]["mean"])

    report += f"- **Best performing category**: {best_category[0].replace('_', ' ').title()} ({best_category[1]['mean']:.2f})\n"
    report += f"- **Most challenging category**: {worst_category[0].replace('_', ' ').title()} ({worst_category[1]['mean']:.2f})\n"

    # Find challenging samples
    challenging_samples = find_challenging_samples(current_results, 5)
    report += f"\n### Most Challenging Questions\n"
    for i, sample in enumerate(challenging_samples, 1):
        report += f"{i}. **Score: {sample['evaluation']['score_global']:.1f}** - {sample['question'][:100]}...\n"

    # Find best samples
    best_samples = find_best_samples(current_results, 5)
    report += f"\n### Best Performing Questions\n"
    for i, sample in enumerate(best_samples, 1):
        report += f"{i}. **Score: {sample['evaluation']['score_global']:.1f}** - {sample['question'][:100]}...\n"

    # RAG analysis if applicable
    if "rag" in model_name.lower():
        other_results = [
            res for res in all_results if res["summary"]["model_name"] != model_name
        ]
        if other_results:
            report += "\n## RAG Performance Analysis\n"
            report += analyze_rag_performance(current_results, other_results)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Analysis report saved to: {output_file}")
    return report



def export_results_to_excel(results: Dict[str, Any], output_file: str = None):
    """Export results to Excel with multiple sheets"""
    if output_file is None:
        output_file = os.path.join(RESULTS_DIR, "evaluation_results.xlsx")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Summary sheet
        summary_data = analyze_performance_by_category(results)
        summary_data.to_excel(writer, sheet_name="Summary", index=False)

        # Detailed results
        detailed_data = []
        for result in results["detailed_results"]:
            row = {
                "Sample_ID": result["sample_idx"],
                "Question": result["question"],
                "Global_Score": result["evaluation"]["score_global"],
                "Generation_Time": result["metadata"]["generation_time"],
                "Evaluation_Time": result["metadata"]["evaluation_time"],
            }

            # Add category scores
            for category, score in result["evaluation"]["scores"].items():
                row[f"Score_{category}"] = score

            # Add ground truth
            for category, value in result["ground_truth"].items():
                row[f"GT_{category}"] = value

            detailed_data.append(row)

        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_excel(writer, sheet_name="Detailed_Results", index=False)

        # Challenging samples
        challenging_samples = find_challenging_samples(results, 20)
        challenging_data = []
        for sample in challenging_samples:
            challenging_data.append(
                {
                    "Sample_ID": sample["sample_idx"],
                    "Question": sample["question"],
                    "Global_Score": sample["evaluation"]["score_global"],
                    "Model_Response": (
                        sample["model_response"][:500] + "..."
                        if len(sample["model_response"]) > 500
                        else sample["model_response"]
                    ),
                }
            )

        challenging_df = pd.DataFrame(challenging_data)
        challenging_df.to_excel(writer, sheet_name="Challenging_Samples", index=False)


def main():
    """Main function for analysis utilities"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Les Audits-Affaires evaluation results")
    parser.add_argument("--results-file", type=str, help="Path to results JSON file")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--report", action="store_true", help="Generate analysis report")
    parser.add_argument("--excel", action="store_true", help="Export to Excel")

    args = parser.parse_args()

    # Load results
    results = load_evaluation_results(args.results_file)

    if args.plots:
        print("Generating plots...")
        create_score_distribution_plot(results)
        create_correlation_heatmap(results)

    if args.report:
        print("Generating analysis report...")
        generate_analysis_report(results)

    if args.excel:
        print("Exporting to Excel...")
        export_results_to_excel(results)

    if not any([args.plots, args.report, args.excel]):
        print("No action specified. Use --plots, --report, or --excel")


if __name__ == "__main__":
    main()
