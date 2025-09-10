import sys
import pytest
from pyspark.sql import SparkSession

def get_spark():
    """
    Ensure SparkSession is available on Databricks.
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        # Create a new one (Databricks will attach to cluster)
        spark = (
            SparkSession.builder
            .appName("pytest-enriched")
            .getOrCreate()
        )
    return spark


if __name__ == "__main__":
    # Add project root to sys.path so "src" can be imported
    sys.path.append("/tmp/pei_tests")

    # Make sure Spark is initialized before tests run
    spark = get_spark()
    print(f"✅ Spark session started: {spark.version}")

    # Run pytest from the project root
    exit_code = pytest.main([
        "--rootdir=/tmp/pei_tests",
        "tests",                  # run all tests under /tmp/pei_tests/tests
        "--disable-warnings",
        "-q"
    ])

    sys.exit(exit_code)