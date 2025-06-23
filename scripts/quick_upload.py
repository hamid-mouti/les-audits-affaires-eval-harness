#!/usr/bin/env python3
"""
Quick upload script for evaluation results.
Automatically detects results in common locations and uploads them.

Usage:
    python scripts/quick_upload.py gpt-4o openai
    python scripts/quick_upload.py claude-3 anthropic
    python scripts/quick_upload.py llama-3-8b meta
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to the path to import upload_results
sys.path.append(str(Path(__file__).parent))

from upload_results import ResultsUploader

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/quick_upload.py <model_name> <provider>")
        print("Example: python scripts/quick_upload.py gpt-4o openai")
        sys.exit(1)
    
    model_name = sys.argv[1]
    provider = sys.argv[2]
    
    print(f"🔍 Auto-detecting results for {model_name} ({provider})...")
    
    # Initialize uploader
    uploader = ResultsUploader()
    
    project_root = Path(__file__).parent.parent
    scores = None
    
    # List of common patterns to check
    patterns_to_check = [
        # Root level files
        f"results_{provider}_{model_name}.json",
        f"results_{provider}_{model_name.replace('-', '_')}.json",
        f"results_{model_name}.json",
        f"{model_name}_results.json",
        # Results directory
        f"results/{model_name}",
        f"results/{model_name.replace('-', '_')}",
        f"results/{provider}_{model_name}",
        f"results/{model_name}_{provider}",
    ]
    
    print("📁 Checking common result locations:")
    
    for pattern in patterns_to_check:
        path = project_root / pattern
        print(f"   • {pattern} ... ", end="")
        
        if path.is_file():
            print("✅ Found file!")
            scores = uploader.parse_results_file(str(path), model_name, provider)
            if scores:
                break
        elif path.is_dir():
            print("✅ Found directory!")
            scores = uploader.parse_results_directory(str(path), model_name, provider)
            if scores:
                break
        else:
            print("❌ Not found")
    
    if not scores:
        print(f"\n❌ No results found for {model_name} ({provider})")
        print("\n💡 Available results:")
        
        # Show available results
        results_dir = project_root / "results"
        if results_dir.exists():
            for item in results_dir.iterdir():
                if item.is_dir():
                    print(f"   📁 {item.name}/")
        
        # Show root level result files
        for item in project_root.glob("results_*.json"):
            print(f"   📄 {item.name}")
        
        print(f"\n🔧 Manual usage:")
        print(f"   python scripts/upload_results.py --model_name {model_name} --provider {provider} --results_file <path_to_file>")
        return
    
    print(f"\n📊 Parsed results successfully!")
    print(f"   • Overall Score: {scores['overall']}%")
    print(f"   • Action Requise: {scores['action_requise']}%")
    print(f"   • Délai Legal: {scores['delai_legal']}%")
    print(f"   • Documents: {scores['documents_obligatoires']}%")
    print(f"   • Impact Financier: {scores['impact_financier']}%")
    print(f"   • Conséquences: {scores['consequences_non_conformite']}%")
    
    # Confirm upload
    confirm = input(f"\n❓ Upload results for {model_name} ({provider}) to leaderboard? [y/N]: ")
    
    if confirm.lower() in ['y', 'yes']:
        print("\n🚀 Uploading results...")
        success = uploader.upload_results(model_name, provider, scores)
        
        if success:
            print("\n🎉 Results uploaded successfully!")
            print("📊 View leaderboard: https://huggingface.co/spaces/legmlai/laal-leaderboard")
        else:
            print("\n❌ Upload failed")
    else:
        print("\n⏹️  Upload cancelled")

if __name__ == "__main__":
    main() 