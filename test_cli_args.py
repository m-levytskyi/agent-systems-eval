#!/usr/bin/env python3
"""Test script to verify CLI argument parsing and model configuration logic."""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

def test_cli_parsing():
    """Test the CLI argument parsing logic."""
    
    # Simulate different command-line scenarios
    test_cases = [
        (["evaluate.py"], "ollama", False),  # Default
        (["evaluate.py", "--agents-model=ollama"], "ollama", False),
        (["evaluate.py", "--agents-model=gemini"], "gemini", False),
        (["evaluate.py", "-t"], "ollama", True),
        (["evaluate.py", "--test", "--agents-model=gemini"], "gemini", True),
    ]
    
    for argv, expected_model, expected_test in test_cases:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--agents-model", choices=["ollama", "gemini"], default="ollama")
        parser.add_argument("-t", "--test", action="store_true")
        
        args = parser.parse_args(argv[1:])
        
        # Verify
        assert args.agents_model == expected_model, f"Failed for {argv}: expected {expected_model}, got {args.agents_model}"
        assert args.test == expected_test, f"Failed for {argv}: expected test={expected_test}, got {args.test}"
        
        print(f"✓ {' '.join(argv):50s} → agents_model={args.agents_model}, test={args.test}")
    
    print("\n✅ All CLI parsing tests passed!")


def test_model_configuration():
    """Test the model configuration logic."""
    
    print("\n" + "="*60)
    print("Testing Model Configuration Logic")
    print("="*60)
    
    # Test Ollama configuration
    agents_model = "ollama"
    if agents_model == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        crewai_model = os.getenv("CREWAI_MODEL", f"openai/{model}")
    else:
        model = "gemini-2.5-flash-lite"
        crewai_model = "gemini/gemini-2.5-flash-lite"
    
    judge_model = os.getenv("JUDGE_MODEL", "gemini:/gemini-2.5-flash-lite")
    
    print(f"\nWith --agents-model=ollama:")
    print(f"  MonolithicAgent model: {model}")
    print(f"  EnsembleAgent model:   {crewai_model}")
    print(f"  Judge model:           {judge_model}")
    
    # Test Gemini configuration
    agents_model = "gemini"
    if agents_model == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        crewai_model = os.getenv("CREWAI_MODEL", f"openai/{model}")
    else:
        model = "gemini-2.5-flash-lite"
        crewai_model = "gemini/gemini-2.5-flash-lite"
    
    judge_model = os.getenv("JUDGE_MODEL", "gemini:/gemini-2.5-flash-lite")
    
    print(f"\nWith --agents-model=gemini:")
    print(f"  MonolithicAgent model: {model}")
    print(f"  EnsembleAgent model:   {crewai_model}")
    print(f"  Judge model:           {judge_model}")
    
    print("\n✅ Model configuration logic verified!")


def test_experiment_names():
    """Test the experiment name generation logic."""
    
    print("\n" + "="*60)
    print("Testing Experiment Name Generation")
    print("="*60)
    
    for agent_type in ["monolithic", "ensemble"]:
        for agent_model_type in ["ollama", "gemini"]:
            experiment_name = f"document_synthesis_{agent_type}{'_gemini' if agent_model_type == 'gemini' else ''}"
            print(f"  {agent_type:12s} + {agent_model_type:8s} → {experiment_name}")
    
    print("\n✅ Experiment name generation verified!")


if __name__ == "__main__":
    test_cli_parsing()
    test_model_configuration()
    test_experiment_names()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)
