#!/usr/bin/env python3
"""
Comprehensive test runner for PEI E-commerce project.
Provides options to run different test suites and generate coverage reports.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def install_dependencies():
    """Install required test dependencies"""
    dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-html>=3.1.0",
        "pytest-xdist>=3.0.0",  # For parallel test execution
        "pyspark>=3.3.0",
        "delta-spark>=2.0.0"
    ]
    
    for dep in dependencies:
        cmd = [sys.executable, "-m", "pip", "install", dep]
        if not run_command(cmd, f"Installing {dep}"):
            return False
    return True


def run_basic_tests():
    """Run the original basic test suite"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_enriched_order_details.py",
        "tests/test_agg_profit.py", 
        "tests/test_enriched_customer_products.py",
        "-v"
    ]
    return run_command(cmd, "Basic Test Suite")


def run_extended_tests():
    """Run the extended test suite for comprehensive coverage"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_enriched_order_details_extended.py",
        "tests/test_agg_profit_extended.py",
        "tests/test_enriched_customer_products_extended.py", 
        "-v", "-m", "not slow"
    ]
    return run_command(cmd, "Extended Test Suite")


def run_integration_tests():
    """Run integration tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_integration.py",
        "-v", "-s"  # -s to see print statements
    ]
    return run_command(cmd, "Integration Test Suite")


def run_all_tests():
    """Run all tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v"
    ]
    return run_command(cmd, "Complete Test Suite")


def run_tests_with_coverage():
    """Run all tests with coverage report"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml:coverage.xml",
        "-v"
    ]
    return run_command(cmd, "Tests with Coverage Report")


def run_performance_tests():
    """Run performance and scalability tests"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v", "-m", "performance or slow",
        "--tb=short"
    ]
    return run_command(cmd, "Performance Test Suite")


def run_parallel_tests():
    """Run tests in parallel for faster execution"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-n", "auto",  # Use all available CPUs
        "-v"
    ]
    return run_command(cmd, "Parallel Test Execution")


def generate_html_report():
    """Generate HTML test report"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--html=test_report.html",
        "--self-contained-html",
        "-v"
    ]
    return run_command(cmd, "HTML Test Report Generation")


def main():
    parser = argparse.ArgumentParser(description="PEI E-commerce Test Runner")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install required test dependencies")
    parser.add_argument("--basic", action="store_true",
                       help="Run basic test suite only")
    parser.add_argument("--extended", action="store_true", 
                       help="Run extended test suite only")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true",
                       help="Run all tests with coverage report")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance tests only")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel")
    parser.add_argument("--html-report", action="store_true",
                       help="Generate HTML test report")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests (default)")
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    success = True
    
    if args.install_deps:
        print("Installing test dependencies...")
        success = install_dependencies()
        if not success:
            sys.exit(1)
    
    # If no specific test type is specified, run all tests
    if not any([args.basic, args.extended, args.integration, args.coverage, 
                args.performance, args.parallel, args.html_report]):
        args.all = True
    
    if args.basic:
        success &= run_basic_tests()
    
    if args.extended:
        success &= run_extended_tests()
    
    if args.integration:
        success &= run_integration_tests()
    
    if args.coverage:
        success &= run_tests_with_coverage()
    
    if args.performance:
        success &= run_performance_tests()
    
    if args.parallel:
        success &= run_parallel_tests()
    
    if args.html_report:
        success &= generate_html_report()
    
    if args.all:
        success &= run_all_tests()
    
    if success:
        print(f"\n{'='*60}")
        print("🎉 All requested tests completed successfully!")
        print(f"{'='*60}")
        
        # Print coverage information if coverage files exist
        if os.path.exists("htmlcov/index.html"):
            print(f"\n📊 Coverage report available at: file://{project_root}/htmlcov/index.html")
        
        if os.path.exists("test_report.html"):
            print(f"📋 Test report available at: file://{project_root}/test_report.html")
            
    else:
        print(f"\n{'='*60}")
        print("❌ Some tests failed. Please check the output above.")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
