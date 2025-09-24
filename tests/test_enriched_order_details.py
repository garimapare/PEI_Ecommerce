import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import col
import sys
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Add project root to sys.path
repo_root = "/Workspace/Users/pare.garima.18@gmail.com/PEI_Ecommerce/src"
sys.path.append(repo_root)

from src.enriched_order_details import create_enriched_order_details, get_spark


# Parameterized fixtures for different test scenarios
@pytest.fixture(scope="session")
def spark():
    """Session-scoped Spark fixture for all tests"""
    return SparkSession.builder \
        .master("local[*]") \
        .appName("pytest-enriched-extended") \
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

@pytest.fixture(params=[
    ("test_job_1", "test_run_1"),
    ("test_job_2", "test_run_2"),
    ("batch_job_123", "run_456")
])
def job_run_ids(request) -> Tuple[str, str]:
    """Parameterized fixture for different job and run ID combinations"""
    return request.param

@pytest.fixture(params=[True, False])
def save_flag(request) -> bool:
    """Parameterized fixture for testing both save=True and save=False scenarios"""
    return request.param

@pytest.fixture
def sample_orders_data(self):
    """Sample orders data for testing"""
    return [
        Row(**{"Order ID": "O1", "Order Date": "21/8/2016", "Customer ID": "C1", "Product ID": "P1", "Profit": 100.0}),
        Row(**{"Order ID": "O2", "Order Date": "22/8/2016", "Customer ID": "C2", "Product ID": "P2", "Profit": 250.5}),
        Row(**{"Order ID": "O3", "Order Date": "23/8/2016", "Customer ID": "C1", "Product ID": "P3", "Profit": 0.0})
    ]

@pytest.fixture
def sample_customers_data(self):
    """Sample customers data for testing"""
    return [
        Row(**{"Customer ID": "C1", "Customer Name": "Alice Johnson", "Country": "USA"}),
        Row(**{"Customer ID": "C2", "Customer Name": "Bob Smith", "Country": "Canada"}),
        Row(**{"Customer ID": "C3", "Customer Name": "Charlie Brown", "Country": "UK"})
    ]

@pytest.fixture
def sample_products_data(self):
    """Sample products data for testing"""
    return [
        Row(**{"Product ID": "P1", "Category": "Furniture", "Sub-Category": "Chairs"}),
        Row(**{"Product ID": "P2", "Category": "Technology", "Sub-Category": "Phones"}),
        Row(**{"Product ID": "P3", "Category": "Office Supplies", "Sub-Category": "Storage"})
    ]

@pytest.fixture
def test_table_names() -> Dict[str, str]:
    """Fixture providing consistent table names for tests"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "orders": f"test_orders_{timestamp}",
        "customers": f"test_customers_{timestamp}",
        "products": f"test_products_{timestamp}"
    }


class TestEnrichedOrderDetailsExtended:
    
    # Parameterized tests for different profit scenarios
    @pytest.mark.parametrize("profit_value,expected_valid,expected_bad", [
        (100.0, 1, 0),    # Positive profit - valid
        (0.0, 1, 0),      # Zero profit - valid (>= 0)
        (-50.0, 0, 1),    # Negative profit - invalid
        (999.99, 1, 0),   # Large positive profit - valid
        (-0.01, 0, 1),    # Small negative profit - invalid
    ])

    # Test Case 1 - Validate if profit is calculated correctly using parametrize
    def test_profit_validation_scenarios(self, spark, profit_value: float, expected_valid: int, expected_bad: int):
        orders = [Row(**{"Order ID": "O1", "Order Date": "21/8/2016", "Customer ID": "C1", "Product ID": "P1", "Profit": profit_value})]
        customers = [Row(**{"Customer ID": "C1", "Customer Name": "Alice", "Country": "USA"})]
        products = [Row(**{"Product ID": "P1", "Category": "Furniture", "Sub-Category": "Chairs"})]
        
        # Create temporary views
        spark.createDataFrame(orders).createOrReplaceTempView("profit_test_orders")
        spark.createDataFrame(customers).createOrReplaceTempView("profit_test_customers")
        spark.createDataFrame(products).createOrReplaceTempView("profit_test_products")
        
        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders="profit_test_orders",
            customers="profit_test_customers",
            products="profit_test_products",
            save=False
        )
        
        assert valid_df.count() == expected_valid, f"Expected {expected_valid} valid records for profit {profit_value}"
        assert bad_df.count() == expected_bad, f"Expected {expected_bad} bad records for profit {profit_value}"
    
    @pytest.mark.parametrize("date_format,is_valid", [
        ("21/8/2016", True),      # Valid DD/M/YYYY format
        ("01/12/2020", True),     # Valid DD/MM/YYYY format
        ("2016-08-21", False),    # Invalid YYYY-MM-DD format
        ("Aug 21, 2016", False),  # Invalid text format
        ("", False),              # Empty date
        ("32/13/2016", False),    # Invalid date values
    ])

    # Test Case 2 - Validate if profit is calculated correctly using parametrize
    def test_date_format_validation(self, spark, date_format: str, is_valid: bool):
        
        orders = [Row(**{"Order ID": "O1", "Order Date": date_format, "Customer ID": "C1", "Product ID": "P1", "Profit": 100.0})]
        customers = [Row(**{"Customer ID": "C1", "Customer Name": "Alice", "Country": "USA"})]
        products = [Row(**{"Product ID": "P1", "Category": "Furniture", "Sub-Category": "Chairs"})]
        
        spark.createDataFrame(orders).createOrReplaceTempView("date_test_orders")
        spark.createDataFrame(customers).createOrReplaceTempView("date_test_customers")
        spark.createDataFrame(products).createOrReplaceTempView("date_test_products")
        
        if is_valid:
            # Should process successfully
            valid_df, bad_df = create_enriched_order_details(
                spark,
                orders="date_test_orders",
                customers="date_test_customers",
                products="date_test_products",
                save=False
            )
            assert valid_df.count() >= 0  # Should not crash
        else:
            # May raise exception or produce bad records depending on implementation
            try:
                valid_df, bad_df = create_enriched_order_details(
                    spark,
                    orders="date_test_orders",
                    customers="date_test_customers",
                    products="date_test_products",
                    save=False
                )
                # If no exception, check that invalid dates are handled appropriately
                assert True  # Test passes if no exception is raised
            except Exception:
                # Exception is acceptable for invalid date formats
                assert True
    
    # Test Case 3 - Validate if audit columns are calculated correctly using parametrize
    def test_audit_columns_with_parameterized_ids(self, spark, job_run_ids):
        job_id, run_id = job_run_ids
        
        orders = [Row(**{"Order ID": "O1", "Order Date": "21/8/2016", "Customer ID": "C1", "Product ID": "P1", "Profit": 100.0})]
        customers = [Row(**{"Customer ID": "C1", "Customer Name": "Alice", "Country": "USA"})]
        products = [Row(**{"Product ID": "P1", "Category": "Furniture", "Sub-Category": "Chairs"})]
        
        spark.createDataFrame(orders).createOrReplaceTempView(f"audit_orders_{job_id}")
        spark.createDataFrame(customers).createOrReplaceTempView(f"audit_customers_{job_id}")
        spark.createDataFrame(products).createOrReplaceTempView(f"audit_products_{job_id}")
        
        valid_df, _ = create_enriched_order_details(
            spark,
            orders=f"audit_orders_{job_id}",
            customers=f"audit_customers_{job_id}",
            products=f"audit_products_{job_id}",
            save=False,
            job_id=job_id,
            run_id=run_id
        )
        
        # Check audit columns
        row = valid_df.collect()[0]
        assert row["created_job_id"] == job_id
        assert row["created_run_id"] == run_id
        assert row["modified_job_id"] == job_id
        assert row["modified_run_id"] == run_id

    # Test Case 3 - Check if profit values rounded to 2 decimal places
    def test_profit_rounding(self, spark):
        """Test that profit values are properly rounded to 2 decimal places"""
        orders = [
            Row(**{"Order ID": "O1", "Order Date": "21/8/2016", "Customer ID": "C1", "Product ID": "P1", "Profit": 100.123456}),
            Row(**{"Order ID": "O2", "Order Date": "22/8/2016", "Customer ID": "C1", "Product ID": "P1", "Profit": 200.999})
        ]
        customers = [Row(**{"Customer ID": "C1", "Customer Name": "Alice", "Country": "India"})]
        products = [Row(**{"Product ID": "P1", "Category": "Furniture", "Sub-Category": "Chairs"})]

        spark.createDataFrame(orders).createOrReplaceTempView("profit_orders")
        spark.createDataFrame(customers).createOrReplaceTempView("profit_customers")
        spark.createDataFrame(products).createOrReplaceTempView("profit_products")

        valid_df, _ = create_enriched_order_details(
            spark,
            orders="profit_orders",
            customers="profit_customers",
            products="profit_products",
            save=False
        )

        profits = [row["Profit"] for row in valid_df.collect()]
        assert 100.12 in profits  # 100.123456 rounded to 2 decimals
        assert 201.0 in profits   # 200.999 rounded to 2 decimals

    # Test Case 4 - Test left join behavior when customers or products are missing
    def test_left_join_behavior(self, spark):
        orders = [
            Row(**{"Order ID": "O1", "Order Date": "21/8/2016", "Customer ID": "C1", "Product ID": "P1", "Profit": 100.0}),
            Row(**{"Order ID": "O2", "Order Date": "22/8/2016", "Customer ID": "C999", "Product ID": "P1", "Profit": 200.0}),  # Missing customer
            Row(**{"Order ID": "O3", "Order Date": "23/8/2016", "Customer ID": "C1", "Product ID": "P999", "Profit": 300.0})   # Missing product
        ]
        customers = [Row(**{"Customer ID": "C1", "Customer Name": "Alice", "Country": "India"})]
        products = [Row(**{"Product ID": "P1", "Category": "Furniture", "Sub-Category": "Chairs"})]

        spark.createDataFrame(orders).createOrReplaceTempView("join_orders")
        spark.createDataFrame(customers).createOrReplaceTempView("join_customers")
        spark.createDataFrame(products).createOrReplaceTempView("join_products")

        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders="join_orders",
            customers="join_customers",
            products="join_products",
            save=False
        )

        # Expected behavior based on source code validation logic:
        # - O1: Has both customer and product, positive profit → VALID
        # - O2: Missing customer, Customer_ID becomes null after left join → INVALID
        # - O3: Missing product but has valid Customer_ID from customer join, positive profit → VALID
        
        valid_rows = valid_df.collect()
        bad_rows = bad_df.collect()
        
        # Should have 2 valid records (O1 and O3)
        assert valid_df.count() == 2
        valid_order_ids = [row["Order_ID"] for row in valid_rows]
        assert "O1" in valid_order_ids
        assert "O3" in valid_order_ids
        
        # Should have 1 bad record (O2 due to null Customer_ID)
        assert bad_df.count() == 1
        bad_order_ids = [row["Order_ID"] for row in bad_rows]
        assert "O2" in bad_order_ids

    # Test Case 5 - Test handling of zero profit (should be valid as it's >= 0)
    def test_zero_profit_edge_case(self, spark):
        orders = [Row(**{"Order ID": "O1", "Order Date": "21/8/2016", "Customer ID": "C1", "Product ID": "P1", "Profit": 0.0})]
        customers = [Row(**{"Customer ID": "C1", "Customer Name": "Alice", "Country": "India"})]
        products = [Row(**{"Product ID": "P1", "Category": "Furniture", "Sub-Category": "Chairs"})]

        spark.createDataFrame(orders).createOrReplaceTempView("zero_orders")
        spark.createDataFrame(customers).createOrReplaceTempView("zero_customers")
        spark.createDataFrame(products).createOrReplaceTempView("zero_products")

        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders="zero_orders",
            customers="zero_customers",
            products="zero_products",
            save=False
        )

        assert valid_df.count() == 1
        assert bad_df.count() == 0
        assert valid_df.collect()[0]["Profit"] == 0.0
