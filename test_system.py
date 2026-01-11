"""
Test script to verify the structure and functionality of the agent system.

This script performs dry-run tests without making actual API calls.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import monolithic
        import ensemble
        import evaluate
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_critical_dependencies():
    """Test that all critical dependencies are available."""
    print("\nTesting critical dependencies...")
    missing = []
    
    # Core dependencies
    dependencies = [
        ("mlflow", "MLflow for experiment tracking"),
        ("openai", "OpenAI SDK for LLM compatibility"),
        ("crewai", "CrewAI for agent orchestration"),
        ("litellm", "LiteLLM for CrewAI LLM integration"),
        ("dotenv", "python-dotenv for configuration"),
        ("PyPDF2", "PyPDF2 for document loading"),
        ("bert_score", "BERTScore for evaluation"),
        ("rouge_score", "ROUGE for evaluation"),
    ]
    
    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {description}: {module_name}")
        except ImportError as e:
            print(f"✗ Missing: {module_name} - {description}")
            missing.append(module_name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def test_crewai_llm_compatibility():
    """Test that CrewAI LLM can be initialized without errors."""
    print("\nTesting CrewAI LLM compatibility...")
    try:
        from crewai import LLM
        
        # Test LLM initialization with a simple model string
        # This should not raise ImportError about LiteLLM
        test_model = "openai/gpt-3.5-turbo"
        llm = LLM(model=test_model)
        print(f"✓ CrewAI LLM initialized successfully with model: {test_model}")
        return True
    except ImportError as e:
        if "LiteLLM" in str(e):
            print(f"✗ LiteLLM dependency missing: {e}")
            print("Install with: pip install litellm")
        else:
            print(f"✗ CrewAI import error: {e}")
        return False
    except Exception as e:
        # Other errors are acceptable (e.g., API key issues) as long as LLM class loads
        print(f"✓ CrewAI LLM class loads (runtime error expected without API key: {e})")
        return True


def test_data_files():
    """Test that all required data files exist."""
    print("\nTesting data files...")
    
    required_files = [
        "data/source_documents/doc1_ai_history.pdf",
        "data/source_documents/doc2_ml_fundamentals.pdf",
        "data/source_documents/doc3_ai_ethics.pdf",
        "data/tasks/synthesis_tasks.json"
    ]
    
    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✓ Found: {filepath}")
        else:
            print(f"✗ Missing: {filepath}")
            all_exist = False
    
    return all_exist


def test_task_structure():
    """Test that tasks file has correct structure."""
    print("\nTesting task file structure...")
    try:
        with open("data/tasks/synthesis_tasks.json", "r", encoding="utf-8") as f:
            tasks = json.load(f)
        
        if not isinstance(tasks, list):
            print("✗ Tasks must be a list")
            return False
        
        if len(tasks) == 0:
            print("✗ No tasks found")
            return False
        
        print(f"✓ Found {len(tasks)} tasks")
        
        for i, task in enumerate(tasks):
            required_keys = ["task_id", "task_description"]
            for key in required_keys:
                if key not in task:
                    print(f"✗ Task {i} missing key: {key}")
                    return False
            print(f"  ✓ Task {task['task_id']}: {task['task_description'][:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ Error reading tasks: {e}")
        return False


def test_agent_initialization():
    """Test that agents can be initialized (without API key)."""
    print("\nTesting agent initialization...")
    try:
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b")
        os.environ.setdefault("CREWAI_MODEL", "openai/qwen2.5:7b")
        
        from monolithic import MonolithicAgent
        from ensemble import EnsembleAgent
        
        mono = MonolithicAgent()
        print(f"✓ MonolithicAgent initialized (model: {mono.model})")
        
        # Note: EnsembleAgent now uses CrewAI Flows, so we just check instantiation
        ens = EnsembleAgent()
        print(f"✓ EnsembleAgent initialized (CrewAI model: {ens.model})")
        
        return True
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return False


def test_document_loading():
    """Test that documents can be loaded."""
    print("\nTesting document loading...")
    try:
        from evaluate import load_source_documents, load_tasks
        
        docs = load_source_documents("data/source_documents")
        print(f"✓ Loaded {len(docs)} documents")
        
        for i, doc in enumerate(docs):
            print(f"  ✓ Document {i+1}: {len(doc)} characters")
        
        tasks = load_tasks("data/tasks/synthesis_tasks.json")
        print(f"✓ Loaded {len(tasks)} tasks")
        
        return True
    except Exception as e:
        print(f"✗ Loading error: {e}")
        return False





def test_project_structure():
    """Test overall project structure."""
    print("\nTesting project structure...")
    
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        ".env.example",
        "monolithic.py",
        "ensemble.py",
        "evaluate.py"
    ]
    
    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✓ Found: {filepath}")
        else:
            print(f"✗ Missing: {filepath}")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("="*60)
    print("Agent Systems Evaluation - Test Suite")
    print("="*60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Critical Dependencies", test_critical_dependencies),
        ("Module Imports", test_imports),
        ("CrewAI LLM Compatibility", test_crewai_llm_compatibility),
        ("Data Files", test_data_files),
        ("Task Structure", test_task_structure),
        ("Agent Initialization", test_agent_initialization),
        ("Document Loading", test_document_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
