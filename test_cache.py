#!/usr/bin/env python3
"""Quick test to verify checkpoint/caching works."""

import json
from pathlib import Path

def test_cache():
    """Verify cache directory structure."""
    cache_dirs = [
        Path("data/cache/summaries"),
        Path("data/cache/ensemble_summaries"),
    ]
    
    print("="*60)
    print("Cache System Test")
    print("="*60)
    
    for cache_dir in cache_dirs:
        print(f"\n{cache_dir}:")
        if not cache_dir.exists():
            print("  ✗ Directory does not exist")
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("  ✓ Created directory")
        else:
            print("  ✓ Directory exists")
            
        # Count cached summaries
        cached_files = list(cache_dir.glob("doc_*_summary.json"))
        print(f"  Cached summaries: {len(cached_files)}")
        
        if cached_files:
            # Show first cache file
            first_cache = cached_files[0]
            with open(first_cache, 'r') as f:
                data = json.load(f)
                print(f"  Sample: {first_cache.name}")
                print(f"    Summary length: {len(data.get('summary', ''))} chars")
                print(f"    Tokens used: {data.get('metadata', {}).get('tokens_used', 0)}")
    
    print("\n" + "="*60)
    print("✓ Cache system ready")
    print("="*60)
    print("\nTo clear cache: rm -rf data/cache/summaries/* data/cache/ensemble_summaries/*")

if __name__ == "__main__":
    test_cache()
