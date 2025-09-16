#!/usr/bin/env python3

import requests
import time
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional
import statistics
import concurrent.futures
from pathlib import Path

class WebGLPerformanceTester:
    """Comprehensive performance testing for the WebGL rendering engine."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.sf_bounds = {
            "full": "-122.5,37.7,-122.3,37.8",
            "downtown": "-122.42,37.77,-122.40,37.79",
            "mission": "-122.43,37.74,-122.41,37.76",
            "sunset": "-122.51,37.74,-122.48,37.77"
        }

    async def test_bulk_endpoint_performance(self) -> Dict:
        """Test the WebGL bulk endpoint with various configurations."""
        print("🚀 Testing WebGL Bulk Endpoint Performance")
        print("=" * 60)

        test_configs = [
            {"name": "Small Dataset", "bbox": self.sf_bounds["downtown"], "max_points": 5000, "target_ms": 200},
            {"name": "Medium Dataset", "bbox": self.sf_bounds["mission"], "max_points": 25000, "target_ms": 500},
            {"name": "Large Dataset", "bbox": self.sf_bounds["full"], "max_points": 100000, "target_ms": 1000},
            {"name": "Extra Large", "bbox": None, "max_points": 150000, "target_ms": 1500},
        ]

        results = []

        for config in test_configs:
            print(f"\n📊 {config['name']} ({config['max_points']:,} max points)")

            # Build URL
            url = f"{self.base_url}/webgl/bulk?zoom=14&max_points={config['max_points']}"
            if config['bbox']:
                url += f"&bbox={config['bbox']}"

            # Measure performance over multiple runs
            times = []
            for run in range(3):
                start_time = time.time()

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                load_time = (time.time() - start_time) * 1000
                                times.append(load_time)

                                if run == 0:  # Only log details for first run
                                    total_points = data.get('total_points', 0)
                                    server_time = data.get('performance', {}).get('processing_time_ms', 0)

                                    print(f"   ✓ Points loaded: {total_points:,}")
                                    print(f"   ✓ Server time: {server_time:.1f}ms")
                                    print(f"   ✓ Total time: {load_time:.1f}ms")
                                    print(f"   ✓ Data format: {data.get('data_format', 'unknown')}")

                except Exception as e:
                    print(f"   ✗ Error: {e}")
                    times.append(float('inf'))

            # Calculate statistics
            if times and all(t != float('inf') for t in times):
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                target_met = avg_time <= config['target_ms']

                print(f"   {'✓' if target_met else '✗'} Avg time: {avg_time:.1f}ms (target: {config['target_ms']}ms)")
                print(f"   📈 Range: {min_time:.1f}ms - {max_time:.1f}ms")

                results.append({
                    **config,
                    'avg_time_ms': avg_time,
                    'min_time_ms': min_time,
                    'max_time_ms': max_time,
                    'target_met': target_met,
                    'points_loaded': total_points if 'total_points' in locals() else 0
                })

        return {"bulk_endpoint_tests": results}

    def test_concurrent_load_performance(self) -> Dict:
        """Test performance under concurrent load."""
        print("\n🔥 Testing Concurrent Load Performance")
        print("=" * 60)

        def make_request(session_id: int) -> Dict:
            url = f"{self.base_url}/webgl/bulk?bbox={self.sf_bounds['full']}&zoom=13&max_points=50000"
            start_time = time.time()

            try:
                response = requests.get(url, timeout=10)
                load_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    return {
                        'session_id': session_id,
                        'success': True,
                        'load_time_ms': load_time,
                        'points': data.get('total_points', 0),
                        'server_time_ms': data.get('performance', {}).get('processing_time_ms', 0)
                    }
                else:
                    return {'session_id': session_id, 'success': False, 'error': f"HTTP {response.status_code}"}

            except Exception as e:
                return {'session_id': session_id, 'success': False, 'error': str(e)}

        # Test with different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        concurrent_results = []

        for concurrency in concurrency_levels:
            print(f"\n📡 Testing {concurrency} concurrent requests")

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                start_time = time.time()
                futures = [executor.submit(make_request, i) for i in range(concurrency)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                total_time = (time.time() - start_time) * 1000

            successful = [r for r in results if r.get('success', False)]
            if successful:
                avg_load_time = statistics.mean([r['load_time_ms'] for r in successful])
                total_points = sum([r.get('points', 0) for r in successful])

                print(f"   ✓ Success rate: {len(successful)}/{concurrency} ({len(successful)/concurrency*100:.1f}%)")
                print(f"   ✓ Avg response: {avg_load_time:.1f}ms")
                print(f"   ✓ Total time: {total_time:.1f}ms")
                print(f"   ✓ Total points: {total_points:,}")
                print(f"   ✓ Throughput: {total_points/total_time*1000:.0f} points/sec")

                concurrent_results.append({
                    'concurrency': concurrency,
                    'success_rate': len(successful) / concurrency,
                    'avg_response_ms': avg_load_time,
                    'total_time_ms': total_time,
                    'total_points': total_points,
                    'throughput_points_per_sec': total_points / total_time * 1000
                })

        return {"concurrent_load_tests": concurrent_results}

    def test_memory_efficiency(self) -> Dict:
        """Test memory efficiency with large datasets."""
        print("\n💾 Testing Memory Efficiency")
        print("=" * 60)

        sizes = [10000, 50000, 100000, 150000]
        memory_results = []

        for size in sizes:
            print(f"\n📏 Testing {size:,} points")

            url = f"{self.base_url}/webgl/bulk?zoom=14&max_points={size}"
            start_time = time.time()

            try:
                response = requests.get(url)
                load_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()

                    # Estimate memory usage
                    points = data.get('points', [])
                    if points:
                        # Each point is [lon, lat, size, r, g, b, a] = 7 numbers
                        point_size_bytes = 7 * 8  # 8 bytes per float64
                        estimated_memory_mb = len(points) * point_size_bytes / (1024 * 1024)

                        # JSON overhead estimation
                        json_size_mb = len(response.content) / (1024 * 1024)

                        efficiency = len(points) / max(load_time, 1)  # points per ms

                        print(f"   ✓ Points loaded: {len(points):,}")
                        print(f"   ✓ Load time: {load_time:.1f}ms")
                        print(f"   ✓ JSON size: {json_size_mb:.2f}MB")
                        print(f"   ✓ Est. memory: {estimated_memory_mb:.2f}MB")
                        print(f"   ✓ Efficiency: {efficiency:.1f} points/ms")

                        memory_results.append({
                            'target_points': size,
                            'actual_points': len(points),
                            'load_time_ms': load_time,
                            'json_size_mb': json_size_mb,
                            'estimated_memory_mb': estimated_memory_mb,
                            'efficiency_points_per_ms': efficiency
                        })

            except Exception as e:
                print(f"   ✗ Error: {e}")

        return {"memory_efficiency_tests": memory_results}

    def test_zoom_level_performance(self) -> Dict:
        """Test performance across different zoom levels."""
        print("\n🔍 Testing Zoom Level Performance")
        print("=" * 60)

        zoom_levels = [8, 10, 12, 14, 16]
        zoom_results = []

        for zoom in zoom_levels:
            print(f"\n🎯 Testing zoom level {zoom}")

            url = f"{self.base_url}/webgl/bulk?bbox={self.sf_bounds['full']}&zoom={zoom}&max_points=100000"
            start_time = time.time()

            try:
                response = requests.get(url)
                load_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    points = data.get('total_points', 0)
                    server_time = data.get('performance', {}).get('processing_time_ms', 0)

                    print(f"   ✓ Points: {points:,}")
                    print(f"   ✓ Server: {server_time:.1f}ms")
                    print(f"   ✓ Total: {load_time:.1f}ms")

                    zoom_results.append({
                        'zoom_level': zoom,
                        'points': points,
                        'server_time_ms': server_time,
                        'total_time_ms': load_time
                    })

            except Exception as e:
                print(f"   ✗ Error: {e}")

        return {"zoom_level_tests": zoom_results}

    async def run_comprehensive_tests(self) -> Dict:
        """Run all performance tests."""
        print("🧪 WebGL Rendering Engine - Comprehensive Performance Test")
        print("=" * 80)

        all_results = {}

        # Test bulk endpoint
        bulk_results = await self.test_bulk_endpoint_performance()
        all_results.update(bulk_results)

        # Test concurrent load
        concurrent_results = self.test_concurrent_load_performance()
        all_results.update(concurrent_results)

        # Test memory efficiency
        memory_results = self.test_memory_efficiency()
        all_results.update(memory_results)

        # Test zoom levels
        zoom_results = self.test_zoom_level_performance()
        all_results.update(zoom_results)

        # Generate summary
        summary = self.generate_performance_summary(all_results)
        all_results["summary"] = summary

        return all_results

    def generate_performance_summary(self, results: Dict) -> Dict:
        """Generate a comprehensive performance summary."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_assessment": "UNKNOWN"
        }

        # Check acceptance criteria
        criteria = []

        # Bulk endpoint performance
        if "bulk_endpoint_tests" in results:
            bulk_tests = results["bulk_endpoint_tests"]
            large_dataset_test = next((t for t in bulk_tests if t['name'] == 'Large Dataset'), None)
            if large_dataset_test:
                target_100k_met = large_dataset_test.get('target_met', False)
                criteria.append(("100k+ points load in <1s", target_100k_met))

        # Concurrent performance
        if "concurrent_load_tests" in results:
            concurrent_tests = results["concurrent_load_tests"]
            high_concurrency = next((t for t in concurrent_tests if t['concurrency'] >= 10), None)
            if high_concurrency:
                concurrent_stable = high_concurrency.get('success_rate', 0) >= 0.9
                criteria.append(("Stable under concurrent load", concurrent_stable))

        # Memory efficiency
        if "memory_efficiency_tests" in results:
            memory_tests = results["memory_efficiency_tests"]
            large_memory_test = next((t for t in memory_tests if t['target_points'] >= 100000), None)
            if large_memory_test:
                memory_efficient = large_memory_test.get('estimated_memory_mb', 1000) <= 100
                criteria.append(("Memory usage <100MB for 100k points", memory_efficient))

        # Overall assessment
        passed_criteria = sum(1 for _, passed in criteria if passed)
        total_criteria = len(criteria)

        if total_criteria == 0:
            summary["overall_assessment"] = "INSUFFICIENT_DATA"
        elif passed_criteria == total_criteria:
            summary["overall_assessment"] = "EXCELLENT"
        elif passed_criteria >= total_criteria * 0.8:
            summary["overall_assessment"] = "GOOD"
        elif passed_criteria >= total_criteria * 0.6:
            summary["overall_assessment"] = "ACCEPTABLE"
        else:
            summary["overall_assessment"] = "NEEDS_IMPROVEMENT"

        summary["criteria_results"] = dict(criteria)
        summary["criteria_passed"] = f"{passed_criteria}/{total_criteria}"

        return summary

    def save_results(self, results: Dict, filename: str = "webgl_performance_results.json"):
        """Save test results to file."""
        output_dir = Path("output/performance")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

async def main():
    """Main test execution."""
    tester = WebGLPerformanceTester()

    try:
        # Run comprehensive tests
        results = await tester.run_comprehensive_tests()

        # Print summary
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 80)

        summary = results.get("summary", {})
        print(f"Overall Assessment: {summary.get('overall_assessment', 'UNKNOWN')}")
        print(f"Criteria Passed: {summary.get('criteria_passed', '0/0')}")

        criteria_results = summary.get("criteria_results", {})
        for criterion, passed in criteria_results.items():
            print(f"{'✓' if passed else '✗'} {criterion}")

        # Save results
        tester.save_results(results)

        print(f"\n🎯 WebGL Engine Performance Test Complete!")
        print(f"Access the engine at: http://localhost:8000/webgl")

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

    return True

if __name__ == "__main__":
    asyncio.run(main())