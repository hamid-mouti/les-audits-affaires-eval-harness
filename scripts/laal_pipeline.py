import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Make sure project root is in path so src package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Internal helper that retries failed evaluations
from les_audits_affaires_eval.model_client import EvaluatorClient

# config constants needed for parsing
from les_audits_affaires_eval.config import DETAILED_FILE, SUMMARY_FILE, OUTPUT_FILE

load_dotenv()


class ResultsDatasetManager:
    """Encapsulates read/write operations for results + requests datasets"""

    REQUESTS_DATASET = "legmlai/laal-requests"
    RESULTS_DATASET = "legmlai/laal-results"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HF_TOKEN")
        if self.token:
            self.api = HfApi(token=self.token)
        else:
            self.api = None

    # -------------- REQUESTS -----------------
    def load_requests(self) -> Dataset:
        return load_dataset(self.REQUESTS_DATASET, split="train", token=self.token)

    def update_request_status(self, request_id: str, new_status: str):
        dataset = self.load_requests()
        # Build updated table
        updated = {k: [] for k in dataset.column_names}
        for row in dataset:
            for k in dataset.column_names:
                if row["request_id"] == request_id and k == "request_status":
                    updated[k].append(new_status)
                else:
                    updated[k].append(row[k])
        updated_ds = Dataset.from_dict(updated)
        if self.token:
            updated_ds.push_to_hub(self.REQUESTS_DATASET, token=self.token, commit_message=f"Update {request_id} -> {new_status}")
        print(f"📝  Request {request_id} status → {new_status}")

    # -------------- RESULTS -----------------
    def load_results(self) -> Dataset:
        try:
            return load_dataset(self.RESULTS_DATASET, split="train", token=self.token)
        except Exception:
            # Dataset absent or empty – return an empty skeleton
            return Dataset.from_dict(self._empty_results_schema())

    def _empty_results_schema(self):
        return {
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

    def clear_results_table(self):
        if not self.token:
            raise RuntimeError("HF_TOKEN required to clear results table")
        empty = Dataset.from_dict(self._empty_results_schema())
        empty.push_to_hub(self.RESULTS_DATASET, token=self.token, commit_message="Clear results table")
        print("🗑️  Cleared results dataset")

    def upload_result_entry(self, scores: Dict[str, float], model_name: str, provider: str, request_id: str):
        if not self.token:
            print("❌ HF_TOKEN required to push results")
            return False
        ds = self.load_results()
        # remove previous entry for same model/provider
        df = ds.to_pandas()
        df = df[(df.model_name != model_name) | (df.model_provider != provider)]
        new_row = {
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
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        Dataset.from_pandas(df).push_to_hub(self.RESULTS_DATASET, token=self.token, commit_message=f"Add results for {model_name}")
        print(f"✅ Uploaded results for {model_name}")
        return True


class SummaryDatasetUploader:
    """Re-uses logic from complete_upload_manager.UploadManager to push summary dataset"""

    def __init__(self):
        """Dynamically import `complete_upload_manager.UploadManager` even if it resides
        outside the current package tree (workspace root)."""

        try:
            from complete_upload_manager import UploadManager as _UM  # type: ignore
        except ModuleNotFoundError:
            # Add the workspace root (two parents up) to sys.path then retry
            root_dir = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(root_dir))
            from complete_upload_manager import UploadManager as _UM  # type: ignore

        self._uploader = _UM()

    def upload(self, results_dir: str, model_name: str):
        """Create and push a flattened summary dataset where each score/justification
        category becomes its own column.  This is now the default path, because HF
        datasets had issues with struct columns in Parquet."""

        try:
            # Load detailed results (using helper from complete_upload_manager)
            results = self._uploader.load_results(results_dir)
            detailed = results.get("detailed", [])
            if not detailed:
                print("❌ No detailed data found; skipping summary dataset upload")
                return False

            # ----------------------------------------------
            # Intersect with ground truth dataset sample_ids
            # ----------------------------------------------
            gt_token = os.getenv("HF_TOKEN_LEADERBOARD_RESULTS", os.getenv("HF_TOKEN"))
            try:
                gt_ds = load_dataset("legmlai/les-audits-affaires", split="train", token=gt_token)
                # Build two lookup maps: by sample_idx and by question text (exact)
                gt_by_idx = {}
                gt_by_question = {}
                for row in gt_ds:
                    if "sample_idx" in row:
                        gt_by_idx[int(row["sample_idx"])] = row
                    gt_by_question[row["question"]] = row
            except Exception as e:
                print(f"⚠️  Could not load ground truth dataset for merge: {e}")
                gt_by_idx = None
                gt_by_question = None

            categories = [
                "action_requise",
                "delai_legal",
                "documents_obligatoires",
                "impact_financier",
                "consequences_non_conformite",
            ]

            # Build DataFrame from detailed results
            pred_rows = []
            for itm in detailed:
                if itm.get("response") == "Évaluation échouée":
                    continue

                sample_id = itm.get("sample_id")
                key_val = None
                if sample_id is not None:
                    key_val = str(sample_id)
                else:
                    key_val = str(itm.get("sample_idx", ""))
                scores = itm.get("scores", itm.get("evaluation", {}).get("scores", {}))
                justifs = itm.get(
                    "justifications", itm.get("evaluation", {}).get("justifications", {})
                )

                base = {
                    "sample_id": key_val,
                    "question": itm.get("question", ""),
                    "response": itm.get("response", itm.get("model_response", "")),
                    "score_global_pred": itm.get(
                        "score_global", itm.get("evaluation", {}).get("score_global", 0)
                    ),
                    "evaluation_timestamp": itm.get(
                        "evaluation_timestamp",
                        itm.get("metadata", {}).get("evaluation_timestamp", ""),
                    ),
                }

                for cat in categories:
                    base[f"score_{cat}_pred"] = scores.get(cat, 0)
                    base[f"justification_{cat}_pred"] = justifs.get(cat, "")

                pred_rows.append(base)

            pred_df = pd.DataFrame(pred_rows)

            if gt_by_idx and gt_by_question:
                # Convert gt_by_idx values to DataFrame
                gt_df_idx = pd.DataFrame(list(gt_by_idx.values()))
                # Rename columns to ground_*
                rename_dict_idx = {cat: f"ground_{cat}" for cat in categories}
                gt_df_idx = gt_df_idx.rename(columns=rename_dict_idx)

                # Convert gt_by_question values to DataFrame
                gt_df_question = pd.DataFrame(list(gt_by_question.values()))
                # Rename columns to ground_*
                rename_dict_question = {cat: f"ground_{cat}" for cat in categories}
                gt_df_question = gt_df_question.rename(columns=rename_dict_question)

                # Keep question, reference_article_content, and ground_* columns if present
                extra_cols = []
                if "reference_article_content" in gt_df_question.columns:
                    extra_cols.append("reference_article_content")

                keep_cols = ["question"] + extra_cols + [f"ground_{cat}" for cat in categories]
                if "question" not in gt_df_question.columns:
                    print("⚠️  Ground truth dataset missing 'question' column; cannot join on question")
                    merged = pred_df
                else:
                    gt_df_question = gt_df_question[keep_cols]
                    merged = pred_df.merge(gt_df_question, on="question", how="left")
            else:
                merged = pred_df

            from datasets import Dataset

            ds = Dataset.from_pandas(merged)

            # repo id convention same as original
            dataset_name = model_name.lower().replace("/", "-").replace("_", "-")
            repo_id = f"les-audites-affaires-leadboard/{dataset_name}"

            token = os.getenv("HF_TOKEN_SUMMARY_DATASETS", os.getenv("HF_TOKEN"))
            ds.push_to_hub(
                repo_id,
                token=token,
                commit_message="Add flattened summary dataset",
            )

            # ----------------------- Build dataset card -----------------------
            # Attempt to load the evaluation summary to embed stats
            summary_path = Path(results_dir) / SUMMARY_FILE
            summary = None
            if summary_path.exists():
                try:
                    import json as _json
                    with open(summary_path) as _f:
                        summary = _json.load(_f)
                except Exception as _exc:
                    print(f"⚠️  Could not load evaluation summary for README: {_exc}")

            # YAML front-matter – avoids the Hub warning about missing metadata
            yaml_lines = ["---", f"model_name: {model_name}", "language: fr"]
            if summary:
                yaml_lines.extend(
                    [
                        f"sample_count: {summary.get('sample_count', 0)}",
                        f"global_score_mean: {round(summary.get('global_score_mean', 0), 2)}",
                        f"global_score_std: {round(summary.get('global_score_std', 0), 2)}",
                        f"successful_evaluations: {summary.get('successful_evaluations', 0)}",
                        f"failed_evaluations: {summary.get('failed_evaluations', 0)}",
                    ]
                )
                # Add category means
                for cat in [
                    "action_requise",
                    "delai_legal",
                    "documents_obligatoires",
                    "impact_financier",
                    "consequences_non_conformite",
                ]:
                    cat_mean = summary.get("category_scores", {}).get(cat, {}).get("mean", 0)
                    yaml_lines.append(f"score_{cat}: {round(cat_mean, 2)}")
            yaml_lines.extend([
                "benchmark_dataset: legmlai/les-audits-affaires",
                "leaderboard_url: https://huggingface.co/spaces/legmlai/les-audites-affaires-leadboard",
                "---",
            ])

            front_matter = "\n".join(yaml_lines)

            # Main markdown body
            body_lines = [
                f"# {model_name} – Évaluation Les Audits-Affaires (Aplatie)",
                "\nJeu de données d'évaluation aplati généré automatiquement.",
                "\nCe jeu de données contient les résultats d'évaluation, échantillon par échantillon, du modèle `"
                + model_name
                + "` sur le benchmark [Les Audits-Affaires](https://huggingface.co/datasets/legmlai/les-audits-affaires).",
                "\nPour comparer ce modèle à d'autres, consultez le tableau de bord 👉 https://huggingface.co/spaces/legmlai/les-audites-affaires-leadboard",
            ]

            if summary:
                import json as _json
                body_lines.append("\n## Evaluation Summary\n")
                body_lines.append("```json")
                body_lines.append(_json.dumps(summary, indent=2, ensure_ascii=False))
                body_lines.append("```")

            description = "\n".join([front_matter] + body_lines)

            # Upload/update README.md
            self._uploader.summary_api.upload_file(
                path_or_fileobj=description.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

            print(f"✅ Summary dataset uploaded to {repo_id}")
            return True
        except Exception as exc:
            print(f"❌ Summary dataset upload failed: {exc}")
            return False


class FailedEvaluationRetrier:
    """Minimal retry mechanism – we only mark failures so pipeline can decide."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.detailed_file = results_dir / DETAILED_FILE
        self.evaluator = EvaluatorClient()

    def has_failures(self) -> bool:
        if not self.detailed_file.exists():
            return False
        for line in self.detailed_file.open():
            if "Évaluation échouée" in line:
                return True
        return False

    def retry(self):
        if not self.detailed_file.exists():
            print("ℹ️  No detailed_results.jsonl file – nothing to retry")
            return

        # Load ground-truth dataset for questions/answers
        token = os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_LEADERBOARD_RESULTS")
        try:
            gt_ds = load_dataset("legmlai/les-audits-affaires", split="train", token=token)
            # Build two lookup maps: by sample_idx and by question text (exact)
            gt_by_idx = {}
            gt_by_question = {}
            for row in gt_ds:
                if "sample_idx" in row:
                    gt_by_idx[int(row["sample_idx"])] = row
                gt_by_question[row["question"]] = row
        except Exception as e:
            print(f"⚠️  Could not load ground-truth dataset – retry aborted: {e}")
            return

        updated_lines = []
        re_evaluated = 0
        failed_cases = []

        import jsonlines as jl
        # ----------------- Scan detailed file and collect failed samples -----------------
        with jl.open(self.detailed_file, "r") as reader:
            for sample in reader:
                eval_data = sample.get("evaluation", {})
                fail_cond_global = eval_data.get("score_global", 0) == 0 and any(
                    "échouée" in str(just).lower() for just in eval_data.get("justifications", {}).values()
                )
                fail_cond_resp = sample.get("response") == "Évaluation échouée" or sample.get("model_response") == "Évaluation échouée"

                if fail_cond_global or fail_cond_resp:
                    # Locate ground-truth row
                    gt_row = None
                    sidx = sample.get("sample_idx")
                    if sidx is not None and int(sidx) in gt_by_idx:
                        gt_row = gt_by_idx[int(sidx)]
                    if gt_row is None:
                        gt_row = gt_by_question.get(sample.get("question"))

                    if gt_row is None:
                        updated_lines.append(sample)  # cannot re-evaluate without GT
                    else:
                        failed_cases.append((sample, gt_row))
                else:
                    updated_lines.append(sample)

        total_failed = len(failed_cases)
        if total_failed == 0:
            print("ℹ️  No failed evaluations found to retry")
            return

        print(f"🔄 Found {total_failed} failed evaluations – re-evaluating in batches of 100 …")

        from datetime import datetime as _dt
        import concurrent.futures

        # --------------- Re-evaluate in batches of 100 concurrently ----------------
        for i in range(0, total_failed, 100):
            batch = failed_cases[i : i + 100]

            def _process(item):
                samp, gt = item
                try:
                    new_eval = self.evaluator.evaluate_response(
                        question=samp.get("question", gt.get("question", "")),
                        model_response=samp.get("response") or samp.get("model_response") or "",
                        ground_truth=gt,
                    )
                    samp["evaluation"] = new_eval
                    samp["evaluation_timestamp"] = _dt.now().isoformat()
                    samp["score_global"] = new_eval.get("score_global", 0)
                    samp["scores"] = new_eval.get("scores", {})
                    samp["justifications"] = new_eval.get("justifications", {})
                except Exception as exc:
                    print(
                        f"⚠️  Re-evaluation failed for sample idx={samp.get('sample_idx')}: {exc}"
                    )
                return samp

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(100, len(batch))
            ) as executor:
                futures = [executor.submit(_process, itm) for itm in batch]
                for fut in concurrent.futures.as_completed(futures):
                    updated_lines.append(fut.result())
                    re_evaluated += 1

            print(f"  ✅ Batch {i // 100 + 1}: processed {len(batch)} samples")

        # Write back updated detailed file
        import jsonlines, shutil, datetime as dt
        backup = self.detailed_file.with_suffix(".bak_" + dt.datetime.now().strftime("%Y%m%d%H%M%S"))
        shutil.copy2(self.detailed_file, backup)
        with jsonlines.open(self.detailed_file, mode="w") as writer:
            for itm in updated_lines:
                writer.write(itm)

        print(f"✅ Retried {re_evaluated} failed evaluations and updated detailed results")

        # Recompute summary
        self._recompute_summary(updated_lines)

    def _recompute_summary(self, all_samples):
        # Compute new summary similar to earlier script
        scores = []
        cat_scores = {
            "action_requise": [],
            "delai_legal": [],
            "documents_obligatoires": [],
            "impact_financier": [],
            "consequences_non_conformite": [],
        }

        succ = 0
        for s in all_samples:
            ev = s.get("evaluation", {})
            if ev.get("score_global", 0) > 0:
                succ += 1
                scores.append(ev.get("score_global", 0))
                for k in cat_scores.keys():
                    cat_scores[k].append(ev.get("scores", {}).get(k, 0))

        def stats(arr):
            import numpy as np
            if not arr:
                return {"mean": 0, "std": 0}
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

        summary = {
            "sample_count": len(all_samples),
            "successful_evaluations": succ,
            "failed_evaluations": len(all_samples) - succ,
            "global_score_mean": stats(scores)["mean"],
            "global_score_std": stats(scores)["std"],
            "category_scores": {k: stats(v) for k, v in cat_scores.items()},
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.results_dir / SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)

        print("✅ Summary recomputed after retries")


class EvaluationPipeline:
    """Main orchestrator"""

    def __init__(self, dry_run: bool = False, max_requests: Optional[int] = None):
        self.dry_run = dry_run
        self.max_requests = max_requests
        self.token = os.getenv("HF_TOKEN")
        self.manager = ResultsDatasetManager(self.token)
        self.summary_uploader = SummaryDatasetUploader()

    # -------------------------- Public entrypoints ----------------------------
    def process_requests(self, retry_failures: bool = True):
        requests_ds = self.manager.load_requests()
        pending = [r for r in requests_ds if r["request_status"] in ("pending", "processing", "in_progress")]
        if self.max_requests:
            pending = pending[: self.max_requests]
        print(f"📥 Found {len(pending)} requests to process")
        for req in pending:
            self._process_single_request(req, retry_failures)

    def process_local_results(self, path: str):
        results_dir = Path(path).expanduser()
        if not results_dir.exists():
            print(f"❌ Results path not found: {results_dir}")
            return
        # --- First, retry failed evaluations if any ---
        retrier = FailedEvaluationRetrier(results_dir)
        retrier.retry()

        summary_path = results_dir / SUMMARY_FILE
        if not summary_path.exists():
            print("❌ evaluation_summary.json missing in results directory")
            return
        with open(summary_path) as f:
            summary = json.load(f)
        model_name = summary.get("model_name", results_dir.name)
        provider = summary.get("model_provider", "unknown")
        dummy_request_id = f"local_{uuid.uuid4().hex[:6]}"
        scores = {
            "overall": round(summary.get("global_score_mean", 0), 1),
            "action_requise": round(summary.get("category_scores", {}).get("action_requise", {}).get("mean", 0), 1),
            "delai_legal": round(summary.get("category_scores", {}).get("delai_legal", {}).get("mean", 0), 1),
            "documents_obligatoires": round(summary.get("category_scores", {}).get("documents_obligatoires", {}).get("mean", 0), 1),
            "impact_financier": round(summary.get("category_scores", {}).get("impact_financier", {}).get("mean", 0), 1),
            "consequences_non_conformite": round(summary.get("category_scores", {}).get("consequences_non_conformite", {}).get("mean", 0), 1),
        }
        if not self.dry_run:
            self.manager.upload_result_entry(scores, model_name, provider, dummy_request_id)
            self.summary_uploader.upload(str(results_dir), model_name)
        print("✅ Local results uploaded")

    def clear_results_table(self):
        if self.dry_run:
            print("🔍 DRY RUN: would clear results table")
        else:
            self.manager.clear_results_table()

    def refresh_summary(self, path: str):
        """Recompute evaluation_summary.json (and evaluation_results.json) from detailed file only."""
        results_dir = Path(path).expanduser()
        detailed_file = results_dir / DETAILED_FILE
        if not detailed_file.exists():
            print(f"❌ {DETAILED_FILE} introuvable dans {results_dir}")
            return

        # Charger toutes les lignes
        import jsonlines, json, numpy as np
        all_samples = []
        with jsonlines.open(detailed_file) as reader:
            for itm in reader:
                all_samples.append(itm)

        if not all_samples:
            print("❌ Aucun échantillon trouvé dans le fichier détaillé – abandon")
            return

        # Statistiques
        scores = []
        cat_scores = {
            "action_requise": [],
            "delai_legal": [],
            "documents_obligatoires": [],
            "impact_financier": [],
            "consequences_non_conformite": [],
        }
        succ = 0
        for s in all_samples:
            ev = s.get("evaluation", {})
            if ev.get("score_global", 0) > 0:
                succ += 1
                scores.append(ev.get("score_global", 0))
                for k in cat_scores:
                    cat_scores[k].append(ev.get("scores", {}).get(k, 0))

        def stats(arr):
            if not arr:
                return {"mean": 0, "std": 0}
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

        summary = {
            "sample_count": len(all_samples),
            "successful_evaluations": succ,
            "failed_evaluations": len(all_samples) - succ,
            "global_score_mean": stats(scores)["mean"],
            "global_score_std": stats(scores)["std"],
            "category_scores": {k: stats(v) for k, v in cat_scores.items()},
            "last_updated": datetime.now().isoformat(),
        }

        # Écriture du nouveau résumé
        with open(results_dir / SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)
        print("✅ evaluation_summary.json mis à jour")

        # Mettre aussi à jour evaluation_results.json (si existant)
        output_file = results_dir / OUTPUT_FILE
        if output_file.exists():
            try:
                with open(output_file) as f:
                    data = json.load(f)
                data["summary"] = summary
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)
                print("✅ evaluation_results.json mis à jour")
            except Exception as exc:
                print(f"⚠️  Impossible de mettre à jour {OUTPUT_FILE}: {exc}")

    # -------------------------- Internal methods -----------------------------
    def _process_single_request(self, req: Dict, retry_failures: bool):
        request_id = req["request_id"]
        model_name = req["model_name"]
        provider = req["model_provider"]

        # ------------------- Check required env variables -------------------
        self._ensure_env_variables(provider)

        print(f"\n🚀 Processing request {request_id} – {model_name} ({provider})")
        if not self.dry_run:
            self.manager.update_request_status(request_id, "in_progress")

        # Run evaluation
        results_dir = self._run_evaluation(model_name, provider)
        if results_dir is None:
            print("❌ Evaluation failed – aborting request")
            if not self.dry_run:
                self.manager.update_request_status(request_id, "failed")
            return

        # Retry failures if any
        if retry_failures:
            retrier = FailedEvaluationRetrier(results_dir)
            if retrier.has_failures():
                retrier.retry()

        # Parse scores from summary
        summary_file = results_dir / SUMMARY_FILE
        if not summary_file.exists():
            print("❌ Summary file not found – marking as failed")
            if not self.dry_run:
                self.manager.update_request_status(request_id, "failed")
            return
        with open(summary_file) as f:
            summary = json.load(f)
        scores = {
            "overall": round(summary.get("global_score_mean", 0), 1),
            "action_requise": round(summary.get("category_scores", {}).get("action_requise", {}).get("mean", 0), 1),
            "delai_legal": round(summary.get("category_scores", {}).get("delai_legal", {}).get("mean", 0), 1),
            "documents_obligatoires": round(summary.get("category_scores", {}).get("documents_obligatoires", {}).get("mean", 0), 1),
            "impact_financier": round(summary.get("category_scores", {}).get("impact_financier", {}).get("mean", 0), 1),
            "consequences_non_conformite": round(summary.get("category_scores", {}).get("consequences_non_conformite", {}).get("mean", 0), 1),
        }
        # Upload results + summary dataset
        if not self.dry_run:
            self.manager.upload_result_entry(scores, model_name, provider, request_id)
            self.summary_uploader.upload(str(results_dir), model_name)
            self.manager.update_request_status(request_id, "finished")
        print(f"🎉 Completed request {request_id}")

    def _run_evaluation(self, model_name: str, provider: str) -> Optional[Path]:
        """Run evaluation harness and return path to results directory."""
        # Prepare env so evaluator writes to unique dir
        safe_name = model_name.replace("/", "_").replace("-", "_")
        results_dir = PROJECT_ROOT / "results" / safe_name
        os.environ["RESULTS_DIR"] = str(results_dir)
        os.environ["MODEL_NAME"] = model_name
        # External model provider config may be needed by user environment
        # We'll forward provider via EXTERNAL_PROVIDER/EXTERNAL_MODEL for remote providers
        if provider.lower() in ("openai", "mistral", "claude", "gemini"):
            os.environ["EXTERNAL_PROVIDER"] = provider.lower()
            os.environ["EXTERNAL_MODEL"] = model_name
        else:
            os.environ.pop("EXTERNAL_PROVIDER", None)
            # user should set MODEL_ENDPOINT for local models

        # NOTE: LesAuditsAffairesEvaluator is imported inside _run_evaluation after env vars are set,
        # to ensure it picks up dynamic RESULT_DIR and MODEL_NAME.
        from les_audits_affaires_eval.evaluator import LesAuditsAffairesEvaluator
        evaluator = LesAuditsAffairesEvaluator()
        try:
            # Run async evaluation fully
            asyncio.run(evaluator.run_evaluation())
            return Path(str(results_dir))
        except Exception as ex:
            print(f"❌ Evaluation error: {ex}")
            return None

    # -------------------------- Utility methods -----------------------------
    def _ensure_env_variables(self, provider: str):
        """Prompt user for missing critical environment variables based on provider."""

        missing: List[str] = []

        # HF token required unless dry-run
        if not self.dry_run and not os.getenv("HF_TOKEN"):
            missing.append("HF_TOKEN")

        # Provider-specific API keys or endpoint
        p = provider.lower()
        if p == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                missing.append("OPENAI_API_KEY")
        elif p == "mistral":
            if not os.getenv("MISTRAL_API_KEY"):
                missing.append("MISTRAL_API_KEY")
        elif p == "claude":
            if not os.getenv("ANTHROPIC_API_KEY"):
                missing.append("ANTHROPIC_API_KEY")
        elif p == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                missing.append("GOOGLE_API_KEY")
        else:
            # Assume local model
            if not os.getenv("MODEL_ENDPOINT"):
                missing.append("MODEL_ENDPOINT")

        if not missing:
            return

        print(
            "❗ Variables d'environnement manquantes : " + ", ".join(missing) + " – saisissez-les maintenant."
        )
        for var in missing:
            try:
                val = input(f"➡️  {var} = ").strip()
                if val:
                    os.environ[var] = val
                else:
                    print(f"⚠️  {var} laissé vide – le traitement risque d'échouer.")
            except EOFError:
                print("Entrée interrompue – abandon.")
                sys.exit(1)


# ------------------------------- CLI ----------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description=(
            "Pipeline unifié d'évaluation Les Audits-Affaires et d'upload.\n\n"
            "Variables d'environnement importantes :\n"
            "  • HF_TOKEN                  → Accès lecture/écriture aux datasets de requêtes et résultats\n"
            "  • MODEL_ENDPOINT            → Endpoint HTTP d'un modèle local\n"
            "  • EXTERNAL_PROVIDER         → openai | mistral | claude | gemini (si modèle hébergé)\n"
            "  • EXTERNAL_MODEL            → Nom/ID du modèle chez le provider\n"
            "  • HF_TOKEN_SUMMARY_DATASETS → (Optionnel) token pour pousser les jeux de données résumé\n"
            "  • HF_TOKEN_LEADERBOARD_RESULTS → (Optionnel) token pour mettre à jour le leaderboard"
        )
    )
    sub = p.add_subparsers(dest="command")

    # process requests
    req_cmd = sub.add_parser("requests", help="Process pending requests (default)")
    req_cmd.add_argument("--max", type=int, help="Maximum requests to process")
    req_cmd.add_argument("--no-retry", action="store_true", help="Do not retry failed evaluations")

    # local processing
    local_cmd = sub.add_parser("local", help="Upload results from local path")
    local_cmd.add_argument("path", help="Path to results directory (containing evaluation_summary.json)")

    # refresh summary
    refresh_cmd = sub.add_parser("refresh", help="Recompute evaluation_summary.json from detailed_results.jsonl")
    refresh_cmd.add_argument("path", help="Path to results directory")

    # clear
    clear_cmd = sub.add_parser("clear-results", help="Clear the results dataset table")

    p.add_argument("--dry-run", action="store_true", help="Run without pushing changes")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_arg_parser().parse_args(argv)
    pipeline = EvaluationPipeline(dry_run=args.dry_run, max_requests=getattr(args, "max", None))

    if args.command in (None, "requests"):
        pipeline.process_requests(retry_failures=not getattr(args, "no_retry", False))

    elif args.command == "local":
        pipeline.process_local_results(args.path)

    elif args.command == "refresh":
        pipeline.refresh_summary(args.path)

    elif args.command == "clear-results":
        pipeline.clear_results_table()
    else:
        print("Unknown command")


if __name__ == "__main__":
    main() 