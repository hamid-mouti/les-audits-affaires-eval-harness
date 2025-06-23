#!/usr/bin/env python3
"""
Script to upload evaluation results to the HuggingFace results dataset.
This script reads results from the /results directory and uploads them to legmlai/laal-results.

Usage:
    python scripts/upload_results.py --model_name "gpt-4o" --provider "openai"
    python scripts/upload_results.py --results_file results_openai_gpt-4o.json
    python scripts/upload_results.py --results_dir results/gpt_4o --model_name "gpt-4o" --provider "openai"
"""

import argparse
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()


class ResultsUploader:
    """Upload evaluation results to HuggingFace dataset"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HF_TOKEN")
        self.results_dataset_name = "legmlai/laal-results"
        self.requests_dataset_name = "legmlai/laal-requests"

        if not self.token:
            print("⚠️  Warning: HF_TOKEN not found. Using public access only.")
            self.api = None
        else:
            self.api = HfApi(token=self.token)
            print("✅ HuggingFace authentication successful")

    def load_results_dataset(self) -> Dataset:
        """Load the results dataset"""
        try:
            if self.token:
                dataset = load_dataset(self.results_dataset_name, token=self.token, split="train")
            else:
                dataset = load_dataset(self.results_dataset_name, split="train")
            print(f"📊 Loaded results dataset with {len(dataset)} entries")
            return dataset
        except Exception as e:
            print(f"❌ Error loading results dataset: {e}")
            # Return empty dataset with correct schema
            return Dataset.from_dict(
                {
                    "result_id": [],
                    "request_id": [],
                    "model_name": [],
                    "model_provider": [],
                    "overall_score": [],
                    "score_action_requise": [],
                    "score_delai_legal": [],
                    "score_documents_obligatoires": [],
                    "score_impact_financier": [],
                    "score_consequences_non_conformite": [],
                    "evaluation_timestamp": [],
                    "is_published": [],
                }
            )

    def load_requests_dataset(self) -> Dataset:
        """Load the requests dataset to find matching request_id"""
        try:
            if self.token:
                dataset = load_dataset(self.requests_dataset_name, token=self.token, split="train")
            else:
                dataset = load_dataset(self.requests_dataset_name, split="train")
            print(f"📝 Loaded requests dataset with {len(dataset)} entries")
            return dataset
        except Exception as e:
            print(f"❌ Error loading requests dataset: {e}")
            return Dataset.from_dict(
                {
                    "request_id": [],
                    "model_name": [],
                    "model_provider": [],
                    "request_type": [],
                    "request_status": [],
                    "contact_email": [],
                    "request_timestamp": [],
                }
            )

    def find_request_id(self, model_name: str, provider: str) -> Optional[str]:
        """Find request_id for a model from the requests dataset"""
        requests_dataset = self.load_requests_dataset()

        for request in requests_dataset:
            if (
                request["model_name"].lower() == model_name.lower()
                and request["model_provider"].lower() == provider.lower()
            ):
                return request["request_id"]

        # If not found, create a new request entry
        print(
            f"⚠️  No matching request found for {model_name} ({provider}). Creating new request..."
        )
        return self.create_request_entry(model_name, provider)

    def create_request_entry(self, model_name: str, provider: str) -> str:
        """Create a new request entry and return the request_id"""
        if not self.token:
            print("❌ Cannot create request entry without HF_TOKEN")
            return f"auto_{uuid.uuid4().hex[:8]}"

        try:
            requests_dataset = self.load_requests_dataset()

            new_request_id = f"req_{uuid.uuid4().hex[:8]}"
            new_request = {
                "request_id": new_request_id,
                "model_name": model_name,
                "model_provider": provider,
                "request_type": "open_source",  # Default assumption
                "request_status": "completed",
                "contact_email": "auto-generated@legml.ai",
                "request_timestamp": datetime.now().isoformat(),
            }

            # Add the new request to the dataset
            updated_data = {
                key: requests_dataset[key] + [new_request[key]]
                for key in requests_dataset.column_names
            }
            updated_dataset = Dataset.from_dict(updated_data)

            # Push to hub
            updated_dataset.push_to_hub(
                self.requests_dataset_name,
                token=self.token,
                commit_message=f"Auto-add request for {model_name} ({provider})",
            )

            print(f"✅ Created new request entry: {new_request_id}")
            return new_request_id

        except Exception as e:
            print(f"❌ Error creating request entry: {e}")
            return f"auto_{uuid.uuid4().hex[:8]}"

    def parse_results_file(self, file_path: str, model_name: str, provider: str) -> Dict:
        """Parse a results JSON file and extract evaluation metrics"""
        print(f"📖 Parsing results file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data:
                raise ValueError("Empty results file")

            # Calculate average scores across all samples
            total_samples = len(data)
            scores = {
                "overall": 0,
                "action_requise": 0,
                "delai_legal": 0,
                "documents_obligatoires": 0,
                "impact_financier": 0,
                "consequences_non_conformite": 0,
            }

            for sample in data:
                if "evaluation" in sample and "scores" in sample["evaluation"]:
                    eval_scores = sample["evaluation"]["scores"]
                    scores["overall"] += sample["evaluation"].get("score_global", 0)
                    scores["action_requise"] += eval_scores.get("action_requise", 0)
                    scores["delai_legal"] += eval_scores.get("delai_legal", 0)
                    scores["documents_obligatoires"] += eval_scores.get("documents_obligatoires", 0)
                    scores["impact_financier"] += eval_scores.get("impact_financier", 0)
                    scores["consequences_non_conformite"] += eval_scores.get(
                        "consequences_non_conformite", 0
                    )

            # Calculate averages
            avg_scores = {key: round(value / total_samples, 1) for key, value in scores.items()}

            print(f"📊 Calculated average scores: {avg_scores}")
            return avg_scores

        except Exception as e:
            print(f"❌ Error parsing results file {file_path}: {e}")
            return None

    def parse_results_directory(self, results_dir: str, model_name: str, provider: str) -> Dict:
        """Parse results from a directory (e.g., results/gpt_4o/)"""
        print(f"📁 Parsing results directory: {results_dir}")

        results_path = Path(results_dir)

        # Look for different result file formats
        json_files = list(results_path.glob("*.json"))
        jsonl_files = list(results_path.glob("*.jsonl"))

        if json_files:
            # Try the main evaluation results file
            for json_file in json_files:
                if "results" in json_file.name.lower():
                    return self.parse_results_file(str(json_file), model_name, provider)

            # If no results file found, try the first JSON file
            return self.parse_results_file(str(json_files[0]), model_name, provider)

        elif jsonl_files:
            # Parse JSONL format
            return self.parse_jsonl_file(str(jsonl_files[0]), model_name, provider)

        else:
            print(f"❌ No results files found in {results_dir}")
            return None

    def parse_jsonl_file(self, file_path: str, model_name: str, provider: str) -> Dict:
        """Parse a JSONL results file"""
        print(f"📖 Parsing JSONL file: {file_path}")

        try:
            scores = {
                "overall": 0,
                "action_requise": 0,
                "delai_legal": 0,
                "documents_obligatoires": 0,
                "impact_financier": 0,
                "consequences_non_conformite": 0,
            }

            total_samples = 0

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if "evaluation" in data and "scores" in data["evaluation"]:
                        eval_scores = data["evaluation"]["scores"]
                        scores["overall"] += data["evaluation"].get("score_global", 0)
                        scores["action_requise"] += eval_scores.get("action_requise", 0)
                        scores["delai_legal"] += eval_scores.get("delai_legal", 0)
                        scores["documents_obligatoires"] += eval_scores.get(
                            "documents_obligatoires", 0
                        )
                        scores["impact_financier"] += eval_scores.get("impact_financier", 0)
                        scores["consequences_non_conformite"] += eval_scores.get(
                            "consequences_non_conformite", 0
                        )
                        total_samples += 1

            if total_samples == 0:
                raise ValueError("No valid samples found in JSONL file")

            # Calculate averages
            avg_scores = {key: round(value / total_samples, 1) for key, value in scores.items()}

            print(f"📊 Calculated average scores from {total_samples} samples: {avg_scores}")
            return avg_scores

        except Exception as e:
            print(f"❌ Error parsing JSONL file {file_path}: {e}")
            return None

    def upload_results(
        self, model_name: str, provider: str, scores: Dict, request_id: Optional[str] = None
    ) -> bool:
        """Upload results to the HuggingFace dataset"""
        if not self.token:
            print("❌ Cannot upload results without HF_TOKEN")
            return False

        try:
            # Load existing results dataset
            results_dataset = self.load_results_dataset()

            # Find or create request_id
            if not request_id:
                request_id = self.find_request_id(model_name, provider)

            # Check if results already exist for this model
            existing_results = [
                r
                for r in results_dataset
                if r["model_name"] == model_name and r["model_provider"] == provider
            ]

            if existing_results:
                print(f"⚠️  Results already exist for {model_name} ({provider}). Updating...")
                # Update existing entry
                for i, result in enumerate(results_dataset):
                    if result["model_name"] == model_name and result["model_provider"] == provider:
                        results_dataset = results_dataset.map(
                            lambda x, idx: (
                                {
                                    **x,
                                    "overall_score": scores["overall"],
                                    "score_action_requise": scores["action_requise"],
                                    "score_delai_legal": scores["delai_legal"],
                                    "score_documents_obligatoires": scores[
                                        "documents_obligatoires"
                                    ],
                                    "score_impact_financier": scores["impact_financier"],
                                    "score_consequences_non_conformite": scores[
                                        "consequences_non_conformite"
                                    ],
                                    "evaluation_timestamp": datetime.now().isoformat(),
                                    "is_published": True,
                                }
                                if idx == i
                                else x
                            ),
                            with_indices=True,
                        )
                        break
            else:
                # Add new entry
                new_result = {
                    "result_id": f"res_{uuid.uuid4().hex[:8]}",
                    "request_id": request_id,
                    "model_name": model_name,
                    "model_provider": provider,
                    "overall_score": scores["overall"],
                    "score_action_requise": scores["action_requise"],
                    "score_delai_legal": scores["delai_legal"],
                    "score_documents_obligatoires": scores["documents_obligatoires"],
                    "score_impact_financier": scores["impact_financier"],
                    "score_consequences_non_conformite": scores["consequences_non_conformite"],
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "is_published": True,
                }

                # Add the new result to the dataset
                updated_data = {
                    key: results_dataset[key] + [new_result[key]]
                    for key in results_dataset.column_names
                }
                results_dataset = Dataset.from_dict(updated_data)

            # Push to hub
            results_dataset.push_to_hub(
                self.results_dataset_name,
                token=self.token,
                commit_message=f"Add/update results for {model_name} ({provider})",
            )

            print(f"✅ Successfully uploaded results for {model_name} ({provider})")
            print(f"📊 Overall Score: {scores['overall']}%")
            return True

        except Exception as e:
            print(f"❌ Error uploading results: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Upload evaluation results to HuggingFace dataset")
    parser.add_argument("--model_name", required=True, help="Model name (e.g., gpt-4o)")
    parser.add_argument("--provider", required=True, help="Model provider (e.g., openai)")
    parser.add_argument("--results_file", help="Path to specific results JSON file")
    parser.add_argument("--results_dir", help="Path to results directory")
    parser.add_argument("--request_id", help="Specific request ID (optional)")
    parser.add_argument("--token", help="HuggingFace token (optional, can use HF_TOKEN env var)")

    args = parser.parse_args()

    print("🚀 Starting results upload process...")
    print(f"Model: {args.model_name} ({args.provider})")

    # Initialize uploader
    uploader = ResultsUploader(token=args.token)

    # Parse results
    scores = None

    if args.results_file:
        if not os.path.exists(args.results_file):
            print(f"❌ Results file not found: {args.results_file}")
            return
        scores = uploader.parse_results_file(args.results_file, args.model_name, args.provider)

    elif args.results_dir:
        if not os.path.exists(args.results_dir):
            print(f"❌ Results directory not found: {args.results_dir}")
            return
        scores = uploader.parse_results_directory(args.results_dir, args.model_name, args.provider)

    else:
        # Auto-detect results
        project_root = Path(__file__).parent.parent

        # Try root-level results file first
        root_results_file = (
            project_root / f"results_{args.provider}_{args.model_name.replace('-', '_')}.json"
        )
        if root_results_file.exists():
            scores = uploader.parse_results_file(
                str(root_results_file), args.model_name, args.provider
            )
        else:
            # Try results directory
            results_dir = project_root / "results" / args.model_name.replace("-", "_")
            if results_dir.exists():
                scores = uploader.parse_results_directory(
                    str(results_dir), args.model_name, args.provider
                )
            else:
                print(
                    f"❌ No results found for {args.model_name}. Please specify --results_file or --results_dir"
                )
                return

    if not scores:
        print("❌ Failed to parse results")
        return

    # Upload results
    success = uploader.upload_results(args.model_name, args.provider, scores, args.request_id)

    if success:
        print("🎉 Results upload completed successfully!")
        print("📊 Check the leaderboard at: https://huggingface.co/spaces/legmlai/laal-leaderboard")
    else:
        print("❌ Results upload failed")


if __name__ == "__main__":
    main()
