#!/usr/bin/env python3

import requests
import time
import json
from typing import Dict, List

def test_performance_targets():
    """Test that the progressive loading system meets performance requirements."""

    base_url = "http://localhost:8000"
    sf_bbox = "-122.5,37.7,-122.3,37.8"

    results = []

    # Test different zoom levels
    test_cases = [
        {"zoom": 3, "name": "Low zoom (hexagonal)", "target_ms": 1000, "max_size_kb": 1},
        {"zoom": 6, "name": "Medium-low zoom (hexagonal)", "target_ms": 1000, "max_size_kb": 1},
        {"zoom": 10, "name": "Medium zoom (clusters)", "target_ms": 1000, "max_size_kb": 10},
        {"zoom": 14, "name": "High zoom (buildings)", "target_ms": 1000, "max_size_kb": 100},
    ]

    print("🚀 Testing Progressive Data Loading Performance")
    print("=" * 60)

    for case in test_cases:
        print(f"\n📊 {case['name']} (Zoom {case['zoom']})")

        # First request (cold)
        url = f"{base_url}/buildings?zoom={case['zoom']}&bbox={sf_bbox}"
        start_time = time.time()

        try:
            response = requests.get(url, timeout=5)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()

                # Extract metrics
                data_type = data.get('type', 'unknown')
                features = data.get('total_features', 0)
                server_time = data.get('performance', {}).get('processing_time_ms', 0)
                estimated_size = data.get('data_size_estimate_kb', 0)

                # Performance checks
                time_ok = server_time < case['target_ms']
                size_ok = estimated_size < case['max_size_kb']

                print(f"   ✓ Data type: {data_type}")
                print(f"   ✓ Features: {features:,}")
                print(f"   {'✓' if time_ok else '✗'} Response time: {response_time:.1f}ms (server: {server_time:.1f}ms)")
                print(f"   {'✓' if size_ok else '✗'} Estimated size: {estimated_size:.2f} KB")
                print(f"   {'✓' if time_ok else '✗'} Meets target: {'Yes' if time_ok else 'No'}")

                results.append({
                    'zoom': case['zoom'],
                    'name': case['name'],
                    'type': data_type,
                    'features': features,
                    'response_time_ms': response_time,
                    'server_time_ms': server_time,
                    'size_kb': estimated_size,
                    'time_target_met': time_ok,
                    'size_target_met': size_ok
                })

                # Test caching (second request)
                print(f"   🔄 Testing cache...")
                start_time = time.time()
                cache_response = requests.get(url, timeout=5)
                cache_time = (time.time() - start_time) * 1000

                if cache_response.status_code == 200:
                    cache_data = cache_response.json()
                    cache_server_time = cache_data.get('performance', {}).get('processing_time_ms', 0)
                    print(f"   ✓ Cache response: {cache_time:.1f}ms (server: {cache_server_time:.1f}ms)")

            else:
                print(f"   ✗ Request failed: {response.status_code}")

        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    time_passes = sum(1 for r in results if r['time_target_met'])
    size_passes = sum(1 for r in results if r['size_target_met'])

    print(f"Total tests: {total_tests}")
    print(f"Time targets met: {time_passes}/{total_tests} ({time_passes/total_tests*100:.1f}%)")
    print(f"Size targets met: {size_passes}/{total_tests} ({size_passes/total_tests*100:.1f}%)")

    # Acceptance criteria check
    print("\n🎯 ACCEPTANCE CRITERIA")
    print("-" * 40)

    criteria = [
        ("Initial map loads in <1 second", all(r['server_time_ms'] < 1000 for r in results)),
        ("Progressive enhancement by zoom level", len(set(r['type'] for r in results)) >= 3),
        ("Hexagonal aggregation for low zoom", any(r['type'] == 'hexagonal' for r in results)),
        ("Statistical clusters for medium zoom", any(r['type'] == 'statistical_clusters' for r in results)),
        ("Individual buildings for high zoom", any(r['type'] == 'individual_buildings' for r in results)),
        ("Data size optimization", all(r['size_target_met'] for r in results)),
    ]

    for criterion, passed in criteria:
        print(f"{'✓' if passed else '✗'} {criterion}")

    overall_pass = all(passed for _, passed in criteria)
    print(f"\n{'🎉 ALL ACCEPTANCE CRITERIA MET!' if overall_pass else '⚠️  Some criteria need attention'}")

    return results

def test_map_interface():
    """Test the enhanced map interface."""
    print("\n🗺️  TESTING MAP INTERFACE")
    print("-" * 40)

    try:
        response = requests.get("http://localhost:8000/map", timeout=5)
        if response.status_code == 200:
            content = response.text

            features = [
                ("Loading indicators", "loading-indicator" in content),
                ("Performance monitoring", "performance-info" in content),
                ("Client-side caching", "dataCache" in content),
                ("Smart prefetching", "prefetchData" in content),
                ("Hexagonal display", "updateHexagonalDisplay" in content),
                ("Cluster display", "updateClusterDisplay" in content),
            ]

            for feature, present in features:
                print(f"{'✓' if present else '✗'} {feature}")

            all_features = all(present for _, present in features)
            print(f"\n{'✓ Map interface enhanced successfully' if all_features else '⚠️  Some features missing'}")

        else:
            print(f"✗ Map interface not accessible: {response.status_code}")

    except Exception as e:
        print(f"✗ Error testing map interface: {e}")

if __name__ == "__main__":
    print("🧪 Progressive Data Loading Performance Test")
    print("=" * 60)

    # Test API performance
    results = test_performance_targets()

    # Test map interface
    test_map_interface()

    print("\n" + "=" * 60)
    print("✅ Performance testing complete!")
    print("\nTo view the interactive map: http://localhost:8000/map")