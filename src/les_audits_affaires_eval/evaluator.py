"""
Main evaluation harness for Les Audits-Affaires LLM benchmark
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import jsonlines
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from .config import *
from .model_client import ChatModelClient, EvaluatorClient, ModelClient, StrictChatModelClient

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LesAuditsAffairesEvaluator:
    """Main evaluator class for the Les Audits-Affaires benchmark"""

    def __init__(self, use_chat_endpoint: bool = False, use_strict_mode: bool = False):
        self.model_client = None
        self.evaluator_client = EvaluatorClient()
        self.results = []
        self.use_chat_endpoint = use_chat_endpoint
        self.use_strict_mode = use_strict_mode

        # Ensure results directory exists
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def load_dataset(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load the Les Audits-Affaires dataset"""
        logger.info(f"Loading dataset: {DATASET_NAME}")

        try:
            dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

            # Convert to list and limit samples if specified
            data = list(dataset)
            if max_samples and max_samples < len(data):
                data = data[:max_samples]
                logger.info(f"Limited dataset to {max_samples} samples")

            logger.info(f"Loaded {len(data)} samples from dataset")
            return data

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    async def evaluate_single_sample(
        self, sample: Dict[str, Any], sample_idx: int
    ) -> Dict[str, Any]:
        """Evaluate a single sample from the dataset (async version)"""
        question = sample["question"]

        # Extract ground truth
        ground_truth = {
            "action_requise": sample.get("action_requise", ""),
            "delai_legal": sample.get("delai_legal", ""),
            "documents_obligatoires": sample.get("documents_obligatoires", ""),
            "impact_financier": sample.get("impact_financier", ""),
            "consequences_non_conformite": sample.get("consequences_non_conformite", ""),
        }

        logger.info(f"Evaluating sample {sample_idx}: {question[:100]}...")

        try:
            # Generate response from the model being evaluated
            start_time = time.time()
            model_response = await self.model_client.generate_response(question)
            generation_time = time.time() - start_time

            logger.debug(f"Model response for sample {sample_idx}: {model_response[:200]}...")

            # Evaluate the response using Azure OpenAI
            eval_start_time = time.time()
            evaluation = self.evaluator_client.evaluate_response(
                question, model_response, ground_truth
            )
            evaluation_time = time.time() - eval_start_time

            # Compile result
            result = {
                "sample_idx": sample_idx,
                "question": question,
                "ground_truth": ground_truth,
                "model_response": model_response,
                "evaluation": evaluation,
                "metadata": {
                    "generation_time": generation_time,
                    "evaluation_time": evaluation_time,
                    "total_time": generation_time + evaluation_time,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            logger.info(
                f"Sample {sample_idx} completed - Global Score: {evaluation.get('score_global', 0)}"
            )
            return result

        except Exception as e:
            logger.error(f"Error evaluating sample {sample_idx}: {e}")
            # Return error result
            return {
                "sample_idx": sample_idx,
                "question": question,
                "ground_truth": ground_truth,
                "model_response": f"ERROR: {str(e)}",
                "evaluation": {
                    "score_global": 0,
                    "scores": {
                        "action_requise": 0,
                        "delai_legal": 0,
                        "documents_obligatoires": 0,
                        "impact_financier": 0,
                        "consequences_non_conformite": 0,
                    },
                    "justifications": {
                        "action_requise": f"Erreur: {str(e)}",
                        "delai_legal": f"Erreur: {str(e)}",
                        "documents_obligatoires": f"Erreur: {str(e)}",
                        "impact_financier": f"Erreur: {str(e)}",
                        "consequences_non_conformite": f"Erreur: {str(e)}",
                    },
                },
                "metadata": {
                    "generation_time": 0,
                    "evaluation_time": 0,
                    "total_time": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                },
            }

    def evaluate_single_sample_sync(
        self, sample: Dict[str, Any], sample_idx: int
    ) -> Dict[str, Any]:
        """Evaluate a single sample from the dataset (sync version)"""
        question = sample["question"]

        # Extract ground truth
        ground_truth = {
            "action_requise": sample.get("action_requise", ""),
            "delai_legal": sample.get("delai_legal", ""),
            "documents_obligatoires": sample.get("documents_obligatoires", ""),
            "impact_financier": sample.get("impact_financier", ""),
            "consequences_non_conformite": sample.get("consequences_non_conformite", ""),
        }

        logger.info(f"Evaluating sample {sample_idx}: {question[:100]}...")

        try:
            # Generate response from the model being evaluated
            start_time = time.time()
            model_response = self.model_client.generate_response_sync(question)
            generation_time = time.time() - start_time

            logger.debug(f"Model response for sample {sample_idx}: {model_response[:200]}...")

            # Evaluate the response using Azure OpenAI
            eval_start_time = time.time()
            evaluation = self.evaluator_client.evaluate_response(
                question, model_response, ground_truth
            )
            evaluation_time = time.time() - eval_start_time

            # Compile result
            result = {
                "sample_idx": sample_idx,
                "question": question,
                "ground_truth": ground_truth,
                "model_response": model_response,
                "evaluation": evaluation,
                "metadata": {
                    "generation_time": generation_time,
                    "evaluation_time": evaluation_time,
                    "total_time": generation_time + evaluation_time,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            logger.info(
                f"Sample {sample_idx} completed - Global Score: {evaluation.get('score_global', 0)}"
            )
            return result

        except Exception as e:
            logger.error(f"Error evaluating sample {sample_idx}: {e}")
            # Return error result
            return {
                "sample_idx": sample_idx,
                "question": question,
                "ground_truth": ground_truth,
                "model_response": f"ERROR: {str(e)}",
                "evaluation": {
                    "score_global": 0,
                    "scores": {
                        "action_requise": 0,
                        "delai_legal": 0,
                        "documents_obligatoires": 0,
                        "impact_financier": 0,
                        "consequences_non_conformite": 0,
                    },
                    "justifications": {
                        "action_requise": f"Erreur: {str(e)}",
                        "delai_legal": f"Erreur: {str(e)}",
                        "documents_obligatoires": f"Erreur: {str(e)}",
                        "impact_financier": f"Erreur: {str(e)}",
                        "consequences_non_conformite": f"Erreur: {str(e)}",
                    },
                },
                "metadata": {
                    "generation_time": 0,
                    "evaluation_time": 0,
                    "total_time": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                },
            }

    async def evaluate_batch(
        self, samples: List[Dict[str, Any]], start_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of samples with controlled concurrency - optimized for high throughput (async version)"""
        # Use a larger semaphore for high-throughput processing
        # But reduce concurrency slightly for longer responses (10K tokens)
        max_concurrent = min(CONCURRENT_REQUESTS, 150)  # Reduced cap for 10K token responses
        semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"Processing batch with {max_concurrent} max concurrent requests (10K token support)"
        )

        async def evaluate_with_semaphore(sample, idx):
            async with semaphore:
                return await self.evaluate_single_sample(sample, start_idx + idx)

        tasks = [evaluate_with_semaphore(sample, idx) for idx, sample in enumerate(samples)]

        # Use tqdm for progress tracking with better batch info
        results = []
        batch_desc = f"Batch {start_idx//BATCH_SIZE + 1} (concurrent: {max_concurrent})"

        for task in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc=batch_desc):
            result = await task
            results.append(result)

            # Save intermediate results
            self.save_intermediate_result(result)

        return results

    def evaluate_batch_sync(
        self, samples: List[Dict[str, Any]], start_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of samples sequentially (sync version)"""
        logger.info(f"Processing batch sequentially (sync mode)")

        results = []
        batch_desc = f"Batch {start_idx//BATCH_SIZE + 1} (sync mode)"

        for idx, sample in enumerate(tqdm(samples, desc=batch_desc)):
            result = self.evaluate_single_sample_sync(sample, start_idx + idx)
            results.append(result)

            # Save intermediate results
            self.save_intermediate_result(result)

        return results

    def save_intermediate_result(self, result: Dict[str, Any]):
        """Save intermediate result to JSONL file"""
        detailed_file_path = os.path.join(RESULTS_DIR, DETAILED_FILE)
        with jsonlines.open(detailed_file_path, mode="a") as writer:
            writer.write(result)

    def load_existing_results(self) -> List[Dict[str, Any]]:
        """Load existing results from detailed results file"""
        detailed_file_path = os.path.join(RESULTS_DIR, DETAILED_FILE)
        results = []

        if not os.path.exists(detailed_file_path):
            return results

        try:
            with jsonlines.open(detailed_file_path, mode="r") as reader:
                for result in reader:
                    results.append(result)
            logger.info(f"Loaded {len(results)} existing results")
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")

        return results

    async def run_evaluation(
        self, max_samples: Optional[int] = None, start_from: int = 0
    ) -> Dict[str, Any]:
        """Run the complete evaluation (async version)"""
        logger.info("Starting Les Audits-Affaires evaluation (async mode)")

        # Load dataset
        full_dataset = self.load_dataset()

        # Slice dataset based on start_from and max_samples
        if start_from > 0:
            logger.info(f"Resuming from sample {start_from}")
            dataset = full_dataset[start_from:]
        else:
            dataset = full_dataset

        if max_samples:
            dataset = dataset[:max_samples]

        logger.info(f"Processing {len(dataset)} samples (starting from index {start_from})")

        # Initialize model client (check for external provider first, then local endpoints)
        external_provider = os.getenv("EXTERNAL_PROVIDER")
        external_model = os.getenv("EXTERNAL_MODEL", "gpt-4o")

        if external_provider:
            # Use external provider (OpenAI, Mistral, Claude, Gemini)
            logger.info(
                f"Using external provider: {external_provider} with model: {external_model}"
            )
            from .clients.external_providers import create_client

            model_client = create_client(external_provider.lower(), model=external_model)
        else:
            # Use local model client (choose between generate, chat, or strict chat endpoint)
            if self.use_strict_mode:
                client_class = StrictChatModelClient
            elif self.use_chat_endpoint:
                client_class = ChatModelClient
            else:
                client_class = ModelClient
            model_client = client_class()

        async with model_client as model_client:
            self.model_client = model_client

            # Only clear previous detailed results if starting from beginning
            detailed_file_path = os.path.join(RESULTS_DIR, DETAILED_FILE)
            if start_from == 0 and os.path.exists(detailed_file_path):
                logger.info("Clearing previous results (starting from beginning)")
                os.remove(detailed_file_path)
            elif start_from > 0:
                logger.info(f"Appending to existing results (resuming from {start_from})")

            # Evaluate in batches
            all_results = []
            total_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
                batch_start_idx = batch_idx * BATCH_SIZE
                batch_end_idx = min(batch_start_idx + BATCH_SIZE, len(dataset))
                batch_samples = dataset[batch_start_idx:batch_end_idx]

                # Adjust sample indices to account for start_from offset
                actual_start_idx = start_from + batch_start_idx
                actual_end_idx = start_from + batch_end_idx - 1

                logger.info(
                    f"Processing batch {batch_idx + 1}/{total_batches} (samples {actual_start_idx}-{actual_end_idx})"
                )

                batch_results = await self.evaluate_batch(batch_samples, actual_start_idx)
                all_results.extend(batch_results)

                # Save progress periodically
                if (batch_idx + 1) % 5 == 0:  # Every 5 batches
                    self.save_progress(all_results, f"batch_{batch_idx+1}_from_{start_from}")

        # Load existing results if resuming
        if start_from > 0:
            logger.info("Loading existing results to compute final metrics")
            existing_results = self.load_existing_results()
            all_results = existing_results + all_results

        # Compile final results
        final_results = self.compute_final_metrics(all_results)

        # Save final results
        self.save_final_results(final_results, all_results)

        return final_results

    def run_evaluation_sync(
        self, max_samples: Optional[int] = None, start_from: int = 0
    ) -> Dict[str, Any]:
        """Run the complete evaluation (sync version)"""
        logger.info("Starting Les Audits-Affaires evaluation (sync mode)")

        # Load dataset
        full_dataset = self.load_dataset()

        # Slice dataset based on start_from and max_samples
        if start_from > 0:
            logger.info(f"Resuming from sample {start_from}")
            dataset = full_dataset[start_from:]
        else:
            dataset = full_dataset

        if max_samples:
            dataset = dataset[:max_samples]

        logger.info(f"Processing {len(dataset)} samples (starting from index {start_from})")

        # Initialize model client (check for external provider first, then local endpoints)
        external_provider = os.getenv("EXTERNAL_PROVIDER")
        external_model = os.getenv("EXTERNAL_MODEL", "gpt-4o")

        if external_provider:
            # Use external provider (OpenAI, Mistral, Claude, Gemini)
            logger.info(
                f"Using external provider: {external_provider} with model: {external_model}"
            )
            from .clients.external_providers import create_client

            self.model_client = create_client(external_provider.lower(), model=external_model)
        else:
            # Use local model client (choose between generate, chat, or strict chat endpoint)
            if self.use_strict_mode:
                from .model_client import StrictChatModelClient

                self.model_client = StrictChatModelClient()
            elif self.use_chat_endpoint:
                from .model_client import ChatModelClient

                self.model_client = ChatModelClient()
            else:
                from .model_client import ModelClient

                self.model_client = ModelClient()

        try:
            # Only clear previous detailed results if starting from beginning
            detailed_file_path = os.path.join(RESULTS_DIR, DETAILED_FILE)
            if start_from == 0 and os.path.exists(detailed_file_path):
                logger.info("Clearing previous results (starting from beginning)")
                os.remove(detailed_file_path)
            elif start_from > 0:
                logger.info(f"Appending to existing results (resuming from {start_from})")

            # Evaluate in batches
            all_results = []
            total_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
                batch_start_idx = batch_idx * BATCH_SIZE
                batch_end_idx = min(batch_start_idx + BATCH_SIZE, len(dataset))
                batch_samples = dataset[batch_start_idx:batch_end_idx]

                # Adjust sample indices to account for start_from offset
                actual_start_idx = start_from + batch_start_idx
                actual_end_idx = start_from + batch_end_idx - 1

                logger.info(
                    f"Processing batch {batch_idx + 1}/{total_batches} (samples {actual_start_idx}-{actual_end_idx})"
                )

                batch_results = self.evaluate_batch_sync(batch_samples, actual_start_idx)
                all_results.extend(batch_results)

                # Save progress periodically
                if (batch_idx + 1) % 5 == 0:  # Every 5 batches
                    self.save_progress(all_results, f"batch_{batch_idx+1}_from_{start_from}")

        finally:
            # Clean up model client if needed
            if hasattr(self.model_client, "session") and self.model_client.session:
                # For sync version, we don't need to close session as it's not used
                pass

        # Load existing results if resuming
        if start_from > 0:
            logger.info("Loading existing results to compute final metrics")
            existing_results = self.load_existing_results()
            all_results = existing_results + all_results

        # Compile final results
        final_results = self.compute_final_metrics(all_results)

        # Save final results
        self.save_final_results(final_results, all_results)

        return final_results

    def compute_final_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute final evaluation metrics"""
        logger.info("Computing final metrics")

        # Extract scores
        global_scores = []
        category_scores = {
            "action_requise": [],
            "delai_legal": [],
            "documents_obligatoires": [],
            "impact_financier": [],
            "consequences_non_conformite": [],
        }

        successful_evaluations = 0
        failed_evaluations = 0

        for result in results:
            evaluation = result["evaluation"]
            if evaluation["score_global"] > 0 or any(
                score > 0 for score in evaluation["scores"].values()
            ):
                successful_evaluations += 1
                global_scores.append(evaluation["score_global"])

                for category, score in evaluation["scores"].items():
                    if category in category_scores:
                        category_scores[category].append(score)
            else:
                failed_evaluations += 1
                # Add zeros for failed evaluations
                global_scores.append(0)
                for category in category_scores:
                    category_scores[category].append(0)

        # Compute statistics
        def compute_stats(scores):
            if not scores:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            return {
                "mean": sum(scores) / len(scores),
                "std": (sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores))
                ** 0.5,
                "min": min(scores),
                "max": max(scores),
            }

        final_metrics = {
            "model_name": MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "sample_count": len(results),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
            "global_score": compute_stats(global_scores),
            "category_scores": {
                category: compute_stats(scores) for category, scores in category_scores.items()
            },
            "configuration": {
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "batch_size": BATCH_SIZE,
                "concurrent_requests": CONCURRENT_REQUESTS,
            },
        }

        logger.info(f"Final global score: {final_metrics['global_score']['mean']:.2f}")
        return final_metrics

    def save_progress(self, results: List[Dict[str, Any]], suffix: str = ""):
        """Save intermediate progress"""
        progress_file = os.path.join(RESULTS_DIR, f"progress_{suffix}.json")
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Progress saved to {progress_file}")

    def save_final_results(
        self, final_metrics: Dict[str, Any], detailed_results: List[Dict[str, Any]]
    ):
        """Save final evaluation results"""

        # Save summary
        summary_file = os.path.join(RESULTS_DIR, SUMMARY_FILE)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)

        # Save complete results
        complete_results = {"summary": final_metrics, "detailed_results": detailed_results}

        results_file = os.path.join(RESULTS_DIR, OUTPUT_FILE)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(complete_results, f, ensure_ascii=False, indent=2)

        # Create CSV summary for easy analysis
        csv_data = []
        for result in detailed_results:
            row = {
                "sample_idx": result["sample_idx"],
                "global_score": result["evaluation"]["score_global"],
                "generation_time": result["metadata"]["generation_time"],
                "evaluation_time": result["metadata"]["evaluation_time"],
            }
            row.update({f"score_{k}": v for k, v in result["evaluation"]["scores"].items()})
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
        df.to_csv(csv_file, index=False)

        logger.info(f"Results saved to:")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Complete: {results_file}")
        logger.info(f"  Detailed: {os.path.join(RESULTS_DIR, DETAILED_FILE)}")
        logger.info(f"  CSV: {csv_file}")


async def main():
    """Main function to run the evaluation"""
    evaluator = LesAuditsAffairesEvaluator()

    try:
        final_results = await evaluator.run_evaluation()

        print("\n" + "=" * 50)
        print("EVALUATION COMPLETED")
        print("=" * 50)
        print(f"Model: {final_results['model_name']}")
        print(f"Dataset: {final_results['dataset_name']}")
        print(f"Samples evaluated: {final_results['sample_count']}")
        print(f"Successful evaluations: {final_results['successful_evaluations']}")
        print(f"Failed evaluations: {final_results['failed_evaluations']}")
        print(
            f"\nGlobal Score: {final_results['global_score']['mean']:.2f} ± {final_results['global_score']['std']:.2f}"
        )
        print("\nCategory Scores:")
        for category, stats in final_results["category_scores"].items():
            print(f"  {category}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print("=" * 50)

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
