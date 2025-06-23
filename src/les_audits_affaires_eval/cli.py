import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .evaluation import LesAuditsAffairesEvaluator

# Setup logging
logger = logging.getLogger(__name__)
from .utils import (
    create_correlation_heatmap,
    create_score_distribution_plot,
    export_results_to_excel,
    generate_analysis_report,
    load_evaluation_results,
)


def _cmd_run(args: argparse.Namespace) -> None:
    """Run the full evaluation based on CLI flags"""
    evaluator = LesAuditsAffairesEvaluator(
        use_chat_endpoint=args.chat,
        use_strict_mode=args.strict,
    )

    try:
        if args.sync:
            # Run synchronous evaluation
            logger.info("Running evaluation in synchronous mode")
            evaluator.run_evaluation_sync(
                max_samples=args.max_samples,
                start_from=args.start_from,
            )
        else:
            # Run asynchronous evaluation (default)
            logger.info("Running evaluation in asynchronous mode")
            asyncio.run(
                evaluator.run_evaluation(
                    max_samples=args.max_samples,
                    start_from=args.start_from,
                )
            )
    except KeyboardInterrupt:
        sys.exit(130)


def _cmd_test_providers(args: argparse.Namespace) -> None:
    """Test external provider connections"""
    print("🏛️ Testing External Provider Connections")
    print("=" * 50)

    # Import here to avoid dependency issues
    try:
        from .clients.external_providers import create_client
    except ImportError as e:
        print(f"❌ Failed to import external providers: {e}")
        return

    async def test_provider(provider_name: str, env_var: str, model: str):
        api_key = os.getenv(env_var)
        if not api_key:
            print(f"⏭️  {provider_name}: No API key ({env_var})")
            return False

        try:
            client_class = create_client(provider_name.lower(), model=model)
            print(f"✅ {provider_name}: Client created successfully")
            return True
        except Exception as e:
            print(f"❌ {provider_name}: Failed - {e}")
            return False

    async def run_tests():
        providers = [
            ("OpenAI", "OPENAI_API_KEY", "gpt-4o"),
            ("Mistral", "MISTRAL_API_KEY", "mistral-large-latest"),
            ("Claude", "ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022"),
            ("Gemini", "GOOGLE_API_KEY", "gemini-1.5-pro"),
        ]

        results = []
        for provider, env_var, model in providers:
            result = await test_provider(provider, env_var, model)
            results.append((provider, result))

        print("\n📊 Summary:")
        available = [p for p, r in results if r]
        unavailable = [p for p, r in results if not r]

        if available:
            print(f"✅ Available: {', '.join(available)}")
        if unavailable:
            print(f"❌ Unavailable: {', '.join(unavailable)}")

        print(f"\n💡 Set API keys in environment variables to enable providers")

    asyncio.run(run_tests())


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze existing evaluation results"""
    results_file = args.results_file
    if not results_file:
        # Try to find results file automatically
        results_dir = Path("results")
        if results_dir.exists():
            for subdir in results_dir.iterdir():
                if subdir.is_dir():
                    potential_file = subdir / "evaluation_results.json"
                    if potential_file.exists():
                        results_file = str(potential_file)
                        break

    if not results_file or not Path(results_file).exists():
        print("❌ No results file found. Use --results-file or run evaluation first.")
        sys.exit(1)

    print(f"📊 Analyzing results from: {results_file}")
    results = load_evaluation_results(results_file)

    if args.report:
        print("📝 Generating analysis report...")
        generate_analysis_report(results)
        print("✅ Report generated")

    if args.plots:
        print("📈 Creating plots...")
        create_score_distribution_plot(results)
        create_correlation_heatmap(results)
        print("✅ Plots created")

    if args.excel:
        print("📋 Exporting to Excel...")
        export_results_to_excel(results)
        print("✅ Excel export completed")


def _cmd_test_evaluator(args: argparse.Namespace) -> None:
    """Test evaluator connection"""
    print("🏛️ Testing Evaluator Connection")
    print("=" * 50)

    try:
        from .model_client import EvaluatorClient

        # Create evaluator instance
        evaluator = EvaluatorClient()
        print(
            f"✅ Evaluator initialized: {evaluator.evaluator_provider} ({evaluator.evaluator_model})"
        )

        # Test with a simple evaluation
        test_question = "Une SARL doit-elle tenir une assemblée générale annuelle?"
        test_response = "Oui, toute SARL doit tenir une assemblée générale annuelle."
        test_ground_truth = {
            "action_requise": "Organiser l'assemblée générale annuelle",
            "delai_legal": "Dans les 6 mois de la clôture de l'exercice",
            "documents_obligatoires": "Comptes annuels, rapport de gestion",
            "impact_financier": "Coûts d'organisation et de convocation",
            "consequences_non_conformite": "Sanctions pénales et dissolution possible",
        }

        print("🧪 Running test evaluation...")
        result = evaluator.evaluate_response(test_question, test_response, test_ground_truth)

        if result and "score_global" in result:
            print(f"✅ Test evaluation successful!")
            print(f"   Score global: {result['score_global']:.1f}/100")
            print(f"   Evaluator working correctly with {evaluator.evaluator_provider}")
        else:
            print("❌ Test evaluation failed - invalid response format")

    except Exception as e:
        print(f"❌ Evaluator test failed: {e}")


def _cmd_info(args: argparse.Namespace) -> None:
    """Show information about the library and configuration"""
    from . import __version__
    from .config import (
        AZURE_OPENAI_ENDPOINT,
        BATCH_SIZE,
        EVALUATOR_ENDPOINT,
        EVALUATOR_MODEL,
        EVALUATOR_PROVIDER,
        MAX_SAMPLES,
        MAX_TOKENS,
        MODEL_ENDPOINT,
        MODEL_NAME,
        TEMPERATURE,
    )

    # Check evaluator configuration
    evaluator_status = "❌ Not configured"
    if EVALUATOR_PROVIDER == "azure" and os.getenv("AZURE_OPENAI_API_KEY"):
        evaluator_status = f"✅ Azure OpenAI ({EVALUATOR_MODEL})"
    elif EVALUATOR_PROVIDER == "openai" and os.getenv("OPENAI_API_KEY"):
        evaluator_status = f"✅ OpenAI ({EVALUATOR_MODEL})"
    elif EVALUATOR_PROVIDER == "mistral" and os.getenv("MISTRAL_API_KEY"):
        evaluator_status = f"✅ Mistral ({EVALUATOR_MODEL})"
    elif EVALUATOR_PROVIDER == "claude" and os.getenv("ANTHROPIC_API_KEY"):
        evaluator_status = f"✅ Claude ({EVALUATOR_MODEL})"
    elif EVALUATOR_PROVIDER == "gemini" and os.getenv("GOOGLE_API_KEY"):
        evaluator_status = f"✅ Gemini ({EVALUATOR_MODEL})"
    elif EVALUATOR_PROVIDER == "local" and EVALUATOR_ENDPOINT:
        evaluator_status = f"✅ Local ({EVALUATOR_ENDPOINT})"

    print(
        f"""
🏛️  Les Audits-Affaires Evaluation Harness v{__version__}
════════════════════════════════════════════════════════

📋 Model Being Evaluated:
  Model Endpoint:     {MODEL_ENDPOINT or 'External Provider'}
  Model Name:         {MODEL_NAME}
  External Provider:  {os.getenv('EXTERNAL_PROVIDER', 'None')}
  
⚖️  Evaluator Configuration:
  Provider:           {EVALUATOR_PROVIDER}
  Model:              {EVALUATOR_MODEL}
  Status:             {evaluator_status}
  
📊 Evaluation Settings:
  Max Samples:        {MAX_SAMPLES}
  Batch Size:         {BATCH_SIZE}
  Temperature:        {TEMPERATURE}
  Max Tokens:         {MAX_TOKENS}

🔌 Available Providers (Model):
  OpenAI:            {'✅' if os.getenv('OPENAI_API_KEY') else '❌'} {'(API key set)' if os.getenv('OPENAI_API_KEY') else '(no API key)'}
  Mistral:           {'✅' if os.getenv('MISTRAL_API_KEY') else '❌'} {'(API key set)' if os.getenv('MISTRAL_API_KEY') else '(no API key)'}
  Claude:            {'✅' if os.getenv('ANTHROPIC_API_KEY') else '❌'} {'(API key set)' if os.getenv('ANTHROPIC_API_KEY') else '(no API key)'}
  Gemini:            {'✅' if os.getenv('GOOGLE_API_KEY') else '❌'} {'(API key set)' if os.getenv('GOOGLE_API_KEY') else '(no API key)'}
  Azure OpenAI:      {'✅' if os.getenv('AZURE_OPENAI_API_KEY') else '❌'} {'(API key set)' if os.getenv('AZURE_OPENAI_API_KEY') else '(no API key)'}

💡 Usage Examples:
  lae-eval run --max-samples 100 --chat       # Async evaluation (default)
  lae-eval run --sync --strict                # Sync evaluation with strict mode
  lae-eval test-providers                      # Test model provider connections
  lae-eval test-evaluator                      # Test evaluator connection
  lae-eval analyze --plots --report           # Generate analysis and plots
  lae-eval info                               # Show this information
    """
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lae-eval",
        description="Les Audits-Affaires – LLM evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lae-eval run --chat --max-samples 50        # Run evaluation with chat endpoint (async)
  lae-eval run --sync --strict                # Run evaluation synchronously with strict mode
  lae-eval run --strict --start-from 100      # Resume from sample 100 with strict mode (async)
  lae-eval test-providers                      # Test external provider connections
  lae-eval analyze --plots --report           # Generate analysis plots and report
  lae-eval analyze --excel results.json       # Export specific results to Excel
  lae-eval info                               # Show configuration info
        """,
    )

    sub = parser.add_subparsers(dest="cmd", required=True, help="Available commands")

    # run command
    run_p = sub.add_parser("run", help="Run an evaluation over the benchmark")
    run_p.add_argument(
        "--chat", action="store_true", help="Use /chat endpoint instead of /generate"
    )
    run_p.add_argument(
        "--strict", action="store_true", help="Strict formatting + repetition handling mode"
    )
    run_p.add_argument("--max-samples", type=int, help="Limit number of samples")
    run_p.add_argument("--start-from", type=int, default=0, help="Dataset index to resume from")
    run_p.add_argument("--sync", action="store_true", help="Run evaluation synchronously")
    run_p.set_defaults(func=_cmd_run)

    # test-providers command
    test_p = sub.add_parser("test-providers", help="Test external provider connections")
    test_p.set_defaults(func=_cmd_test_providers)

    # test-evaluator command
    test_eval_p = sub.add_parser("test-evaluator", help="Test evaluator connection")
    test_eval_p.set_defaults(func=_cmd_test_evaluator)

    # analyze command
    analyze_p = sub.add_parser("analyze", help="Analyze evaluation results")
    analyze_p.add_argument("--results-file", type=str, help="Path to results JSON file")
    analyze_p.add_argument("--plots", action="store_true", help="Generate visualization plots")
    analyze_p.add_argument("--report", action="store_true", help="Generate analysis report")
    analyze_p.add_argument("--excel", action="store_true", help="Export to Excel format")
    analyze_p.set_defaults(func=_cmd_analyze)

    # info command
    info_p = sub.add_parser("info", help="Show library information and configuration")
    info_p.set_defaults(func=_cmd_info)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry-point used by both `python -m` and the console script"""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
