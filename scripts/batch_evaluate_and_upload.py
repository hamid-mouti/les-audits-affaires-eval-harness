#!/usr/bin/env python3
"""
Batch evaluation and upload script.
This script processes all pending requests, evaluates them, and uploads results to the leaderboard.

Usage:
    python scripts/batch_evaluate_and_upload.py
    python scripts/batch_evaluate_and_upload.py --dry-run
    python scripts/batch_evaluate_and_upload.py --max-requests 5
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from upload_results import ResultsUploader

# Load environment variables
load_dotenv()


class BatchEvaluator:
    """Batch process requests, evaluate models, and upload results"""

    def __init__(self, token: Optional[str] = None, dry_run: bool = False):
        self.token = token or os.getenv("HF_TOKEN")
        self.dry_run = dry_run
        self.results_dataset_name = "legmlai/laal-results"
        self.requests_dataset_name = "legmlai/laal-requests"

        if not self.token:
            print("❌ HF_TOKEN required for batch processing")
            sys.exit(1)

        self.api = HfApi(token=self.token)
        self.uploader = ResultsUploader(token=self.token)

        print("✅ HuggingFace authentication successful")
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")

    def load_requests_dataset(self) -> Dataset:
        """Load the requests dataset"""
        try:
            dataset = load_dataset(self.requests_dataset_name, token=self.token, split="train")
            print(f"📝 Loaded requests dataset with {len(dataset)} entries")
            return dataset
        except Exception as e:
            print(f"❌ Error loading requests dataset: {e}")
            sys.exit(1)

    def get_pending_requests(self) -> List[Dict]:
        """Get all pending and processing requests"""
        requests_dataset = self.load_requests_dataset()

        pending_requests = []
        for request in requests_dataset:
            if request["request_status"] in ["pending", "processing"]:
                pending_requests.append(dict(request))

        print(f"⏳ Found {len(pending_requests)} pending/processing requests")
        return pending_requests

    def update_request_status(self, request_id: str, status: str) -> bool:
        """Update request status in the dataset"""
        if self.dry_run:
            print(f"🔍 DRY RUN: Would update {request_id} to {status}")
            return True

        try:
            requests_dataset = self.load_requests_dataset()

            # Update the specific request
            updated_data = {}
            for key in requests_dataset.column_names:
                updated_data[key] = []

            for request in requests_dataset:
                for key in requests_dataset.column_names:
                    if request["request_id"] == request_id and key == "request_status":
                        updated_data[key].append(status)
                    else:
                        updated_data[key].append(request[key])

            updated_dataset = Dataset.from_dict(updated_data)
            updated_dataset.push_to_hub(
                self.requests_dataset_name,
                token=self.token,
                commit_message=f"Update request {request_id} status to {status}",
            )

            print(f"✅ Updated {request_id} status to {status}")
            return True

        except Exception as e:
            print(f"❌ Error updating request {request_id}: {e}")
            return False

    def evaluate_model(self, model_name: str, provider: str) -> Optional[Dict]:
        """
        Evaluate a model and return scores.
        This is a placeholder - you would integrate with your actual evaluation pipeline.
        """
        print(f"🔄 Evaluating {model_name} ({provider})...")

        # Check if results already exist in results directory
        project_root = Path(__file__).parent.parent

        # Common patterns for existing results
        patterns_to_check = [
            f"results_{provider}_{model_name}.json",
            f"results_{provider}_{model_name.replace('-', '_')}.json",
            f"results_{model_name}.json",
            f"results/{model_name}",
            f"results/{model_name.replace('-', '_')}",
            f"results/{provider}_{model_name}",
        ]

        for pattern in patterns_to_check:
            path = project_root / pattern
            if path.is_file():
                print(f"✅ Found existing results file: {pattern}")
                return self.uploader.parse_results_file(str(path), model_name, provider)
            elif path.is_dir():
                print(f"✅ Found existing results directory: {pattern}")
                return self.uploader.parse_results_directory(str(path), model_name, provider)

        # If no existing results found, you would typically:
        # 1. Run your evaluation pipeline here
        # 2. Generate results
        # 3. Save them to a file
        # 4. Parse and return the scores

        print(f"⚠️  No existing results found for {model_name} ({provider})")
        print("💡 To add evaluation capability:")
        print("   1. Integrate with your evaluation pipeline")
        print("   2. Or manually run evaluation and place results in /results directory")

        # For now, return None to indicate evaluation needed
        return None

    def simulate_evaluation(self, model_name: str, provider: str) -> Dict:
        """
        Simulate evaluation with dummy scores for testing.
        Remove this in production and use actual evaluation.
        """
        print(f"🎭 SIMULATING evaluation for {model_name} ({provider})...")
        time.sleep(1)  # Simulate evaluation time

        # Generate realistic but random scores
        import random

        base_score = random.randint(60, 90)

        scores = {
            "overall": base_score + random.randint(-5, 5),
            "action_requise": base_score + random.randint(-10, 10),
            "delai_legal": base_score + random.randint(-10, 10),
            "documents_obligatoires": base_score + random.randint(-10, 10),
            "impact_financier": base_score + random.randint(-10, 10),
            "consequences_non_conformite": base_score + random.randint(-10, 10),
        }

        # Ensure scores are within bounds
        for key in scores:
            scores[key] = max(0, min(100, scores[key]))

        print(f"🎯 Simulated scores: Overall {scores['overall']}%")
        return scores

    def process_all_requests(
        self, max_requests: Optional[int] = None, simulate: bool = False
    ) -> Dict:
        """Process all pending requests"""
        pending_requests = self.get_pending_requests()

        if not pending_requests:
            print("✅ No pending requests to process")
            return {"processed": 0, "successful": 0, "failed": 0}

        if max_requests:
            pending_requests = pending_requests[:max_requests]
            print(f"🔢 Limited to {max_requests} requests")

        results = {"processed": 0, "successful": 0, "failed": 0}

        for i, request in enumerate(pending_requests, 1):
            print(f"\n📋 Processing request {i}/{len(pending_requests)}")
            print(f"   Request ID: {request['request_id']}")
            print(f"   Model: {request['model_name']} ({request['model_provider']})")

            results["processed"] += 1

            # Update status to processing
            if not self.update_request_status(request["request_id"], "processing"):
                print("❌ Failed to update status to processing")
                results["failed"] += 1
                continue

            try:
                # Evaluate the model
                if simulate:
                    scores = self.simulate_evaluation(
                        request["model_name"], request["model_provider"]
                    )
                else:
                    scores = self.evaluate_model(request["model_name"], request["model_provider"])

                if not scores:
                    print(f"❌ No evaluation results for {request['model_name']}")
                    self.update_request_status(request["request_id"], "failed")
                    results["failed"] += 1
                    continue

                # Upload results
                success = False
                if not self.dry_run:
                    success = self.uploader.upload_results(
                        request["model_name"],
                        request["model_provider"],
                        scores,
                        request["request_id"],
                    )
                else:
                    print(f"🔍 DRY RUN: Would upload results for {request['model_name']}")
                    success = True

                if success:
                    self.update_request_status(request["request_id"], "completed")
                    results["successful"] += 1
                    print(f"✅ Successfully processed {request['model_name']}")
                else:
                    self.update_request_status(request["request_id"], "failed")
                    results["failed"] += 1
                    print(f"❌ Failed to upload results for {request['model_name']}")

            except Exception as e:
                print(f"❌ Error processing {request['model_name']}: {e}")
                self.update_request_status(request["request_id"], "failed")
                results["failed"] += 1

        return results

    def clear_all_requests(self) -> bool:
        """Clear all requests from the dataset (use with caution!)"""
        if self.dry_run:
            print("🔍 DRY RUN: Would clear all requests")
            return True

        try:
            # Create empty dataset with correct schema
            empty_dataset = Dataset.from_dict(
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

            empty_dataset.push_to_hub(
                self.requests_dataset_name,
                token=self.token,
                commit_message="Clear all requests for batch processing",
            )

            print("✅ All requests cleared")
            return True

        except Exception as e:
            print(f"❌ Error clearing requests: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate and upload results")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    parser.add_argument("--max-requests", type=int, help="Maximum number of requests to process")
    parser.add_argument(
        "--simulate", action="store_true", help="Use simulated evaluation scores for testing"
    )
    parser.add_argument(
        "--clear-requests",
        action="store_true",
        help="Clear all requests before processing (dangerous!)",
    )
    parser.add_argument("--token", help="HuggingFace token (optional, can use HF_TOKEN env var)")

    args = parser.parse_args()

    print("🚀 Starting batch evaluation and upload process...")

    # Initialize batch evaluator
    evaluator = BatchEvaluator(token=args.token, dry_run=args.dry_run)

    # Clear requests if requested
    if args.clear_requests:
        print("\n⚠️  WARNING: This will clear ALL requests!")
        if not args.dry_run:
            confirm = input("Type 'DELETE ALL' to confirm: ")
            if confirm != "DELETE ALL":
                print("❌ Confirmation failed. Aborting.")
                return

        evaluator.clear_all_requests()

    # Process requests
    print(f"\n📊 Processing pending requests...")
    results = evaluator.process_all_requests(max_requests=args.max_requests, simulate=args.simulate)

    # Summary
    print(f"\n📈 Batch Processing Summary:")
    print(f"   • Processed: {results['processed']}")
    print(f"   • Successful: {results['successful']}")
    print(f"   • Failed: {results['failed']}")

    if results["successful"] > 0:
        print(f"\n🎉 {results['successful']} models successfully evaluated and uploaded!")
        if not args.dry_run:
            print(
                "📊 Check the leaderboard: https://huggingface.co/spaces/legmlai/laal-leaderboard"
            )

    if results["failed"] > 0:
        print(f"\n⚠️  {results['failed']} requests failed to process")


if __name__ == "__main__":
    main()
