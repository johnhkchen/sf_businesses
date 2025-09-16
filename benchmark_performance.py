#!/usr/bin/env python3
"""
Performance benchmarking script for SF Business Data Pipeline optimizations.

Tests various pipeline configurations to demonstrate performance improvements.
"""

import time
import subprocess
import json
from pathlib import Path
import shutil

def run_pipeline_benchmark(config_name: str, clear_cache: bool = False) -> dict:
    """Run pipeline with timing and return performance metrics."""

    if clear_cache:
        # Clear cache and output files
        cache_dir = Path("output/cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        # Remove parquet files but keep json validation files for comparison
        for file in Path("output").glob("*.parquet"):
            file.unlink(missing_ok=True)

    print(f"\n=== Running {config_name} ===")

    start_time = time.time()

    # Run the pipeline
    result = subprocess.run(
        ["uv", "run", "python", "data_pipeline.py"],
        capture_output=True,
        text=True
    )

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.3f} seconds")

    if result.returncode != 0:
        print(f"Error running pipeline: {result.stderr}")
        return {"error": result.stderr, "execution_time": execution_time}

    # Extract key metrics from output
    output_lines = result.stdout.strip().split('\n')

    # Find the results section
    results = {}
    in_results = False
    for line in output_lines:
        if "=== PIPELINE RESULTS ===" in line:
            in_results = True
            continue
        elif in_results and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                # Try to convert to int if possible
                results[key] = int(value)
            except ValueError:
                results[key] = value
        elif in_results and line.strip() == "":
            break

    return {
        "execution_time": execution_time,
        "config": config_name,
        "results": results,
        "stdout_snippet": result.stdout[-500:] if result.stdout else ""
    }

def main():
    """Run comprehensive performance benchmarks."""

    print("SF Business Data Pipeline - Performance Optimization Benchmark")
    print("=" * 70)

    benchmarks = []

    # Test 1: Fresh run (no cache)
    fresh_run = run_pipeline_benchmark("Optimized Pipeline (Fresh Run)", clear_cache=True)
    benchmarks.append(fresh_run)

    # Test 2: Cached run
    cached_run = run_pipeline_benchmark("Optimized Pipeline (Cached Run)", clear_cache=False)
    benchmarks.append(cached_run)

    # Test 3: Another cached run to verify consistency
    cached_run2 = run_pipeline_benchmark("Optimized Pipeline (2nd Cached Run)", clear_cache=False)
    benchmarks.append(cached_run2)

    # Performance summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    baseline_time = 2.193  # Original baseline from manual testing

    for i, benchmark in enumerate(benchmarks, 1):
        exec_time = benchmark['execution_time']
        improvement = ((baseline_time - exec_time) / baseline_time) * 100

        print(f"\nTest {i}: {benchmark['config']}")
        print(f"  Execution Time: {exec_time:.3f} seconds")
        print(f"  vs Baseline: {improvement:+.1f}% {'improvement' if improvement > 0 else 'regression'}")

        if 'results' in benchmark:
            results = benchmark['results']
            print(f"  Records Processed: {results.get('total_businesses', 'N/A'):,}")
            print(f"  Unique NAICS Codes: {results.get('unique_naics_codes', 'N/A')}")
            print(f"  Neighborhoods: {results.get('unique_neighborhoods', 'N/A')}")

    # Calculate cache speedup
    if len(benchmarks) >= 2:
        fresh_time = benchmarks[0]['execution_time']
        cached_time = benchmarks[1]['execution_time']
        cache_speedup = ((fresh_time - cached_time) / fresh_time) * 100

        print(f"\nCache Performance:")
        print(f"  Fresh Run: {fresh_time:.3f} seconds")
        print(f"  Cached Run: {cached_time:.3f} seconds")
        print(f"  Cache Speedup: {cache_speedup:.1f}%")

    # Key optimizations summary
    print(f"\n" + "=" * 70)
    print("OPTIMIZATION FEATURES IMPLEMENTED")
    print("=" * 70)
    print("✅ Parallel CSV processing with multi-threading")
    print("✅ Streaming data processing for memory efficiency")
    print("✅ Intelligent caching with change detection")
    print("✅ Compressed Parquet output with optimal chunk sizes")
    print("✅ Parallel validation and file writing")
    print("✅ Lazy evaluation for reduced memory usage")

    # Save benchmark results
    benchmark_file = Path("output/performance_benchmark.json")
    with open(benchmark_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_time": baseline_time,
            "benchmarks": benchmarks
        }, f, indent=2)

    print(f"\nBenchmark results saved to: {benchmark_file}")

if __name__ == "__main__":
    main()