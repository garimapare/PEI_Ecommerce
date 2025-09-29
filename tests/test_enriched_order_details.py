import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pytest
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import os
import sys
import uuid

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enriched_order_details import create_enriched_order_details, get_spark

# Define global schemas
ORDERS_SCHEMA = StructType([
    StructField("Order ID", StringType(), True),
    StructField("Order Date", StringType(), True),
    StructField("Customer ID", StringType(), True),
    StructField("Product ID", StringType(), True),
    StructField("Profit", DoubleType(), True)
])

CUSTOMERS_SCHEMA = StructType([
    StructField("Customer ID", StringType(), True),
    StructField("Customer Name", StringType(), True),
    StructField("Country", StringType(), True)
])

PRODUCTS_SCHEMA = StructType([
    StructField("Product ID", StringType(), True),
    StructField("Category", StringType(), True),
    StructField("Sub-Category", StringType(), True)
])

# Parameterized fixtures for different test scenarios
@pytest.fixture(scope="session")
def spark():
    # Session-scoped Spark fixture for all tests
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
    # Parameterized fixture for different job and run ID combinations
    return request.param

@pytest.fixture(params=[True, False])
def save_flag(request) -> bool:
    # Parameterized fixture for testing both save=True and save=False scenarios
    return request.param

@pytest.fixture
def sample_orders_data():
    # Sample orders data for testing
    data = [
        {"Order ID": "O1", "Order Date": "21/8/2016", "Customer ID": "C1", "Product ID": "P1", "Profit": 100.0},
        {"Order ID": "O2", "Order Date": "22/8/2016", "Customer ID": "C2", "Product ID": "P2", "Profit": 250.5},
        {"Order ID": "O3", "Order Date": "23/8/2016", "Customer ID": "C1", "Product ID": "P3", "Profit": 0.0}
    ]
    return data

@pytest.fixture
def sample_customers_data():
    # Sample customers data for testing
    data = [
        {"Customer ID": "C1", "Customer Name": "Alice Johnson", "Country": "USA"},
        {"Customer ID": "C2", "Customer Name": "Bob Smith", "Country": "Canada"},
        {"Customer ID": "C3", "Customer Name": "Charlie Brown", "Country": "UK"}
    ]
    return data

@pytest.fixture
def sample_products_data():
    # Sample products data for testing
    data = [
        {"Product ID": "P1", "Category": "Furniture", "Sub-Category": "Chairs"},
        {"Product ID": "P2", "Category": "Technology", "Sub-Category": "Phones"},
        {"Product ID": "P3", "Category": "Office Supplies", "Sub-Category": "Storage"}
    ]
    return data

@pytest.fixture
def test_table_names() -> Dict[str, str]:
    # Fixture providing consistent table names for tests
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "orders": f"test_orders_{timestamp}",
        "customers": f"test_customers_{timestamp}",
        "products": f"test_products_{timestamp}"
    }


class TestEnrichedOrderDetailsExtended:

    # Parameterized tests for different profit scenarios
    @pytest.mark.parametrize("test_scenario,profit_value,expected_valid_count,expected_bad_count", [
        ("positive_profit", 100.0, 1, 0),   # Positive profit - valid
        ("zero_profit", 0.0, 1, 0),         # Zero profit - valid (edge case)
        ("negative_profit", -50.0, 0, 1),   # Negative profit - invalid
        ("large_positive_profit", 999.99, 1, 0),  # Large positive profit - valid
        ("small_negative_profit", -0.01, 0, 1),   # Small negative profit - invalid
    ])

    # Test case 2 - Parameterized test for profit validation
    def test_profit_validation_scenarios(self, spark, test_scenario: str, profit_value: float, 
                                       expected_valid_count: int, expected_bad_count: int):
        # Test data as tuples
        order_id = f"O1_{test_scenario}"  # Unique order ID based on test scenario
        orders_data = [(order_id, "21/8/2016", "C1", "P1", profit_value)]
        customers_data = [("C1", "Alice", "USA")]
        products_data = [("P1", "Furniture", "Chairs")]
        
        # Create DataFrames with explicit schemas
        orders_df = spark.createDataFrame(orders_data, schema=ORDERS_SCHEMA)
        customers_df = spark.createDataFrame(customers_data, schema=CUSTOMERS_SCHEMA)
        products_df = spark.createDataFrame(products_data, schema=PRODUCTS_SCHEMA)
        
        # Call the function under test with DataFrames directly
        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders=orders_df,
            customers=customers_df,
            products=products_df,
            save=False
        )
        
        # Verify counts
        assert valid_df.count() == expected_valid_count, \
            f"Expected {expected_valid_count} valid records for scenario '{test_scenario}'"
        assert bad_df.count() == expected_bad_count, \
            f"Expected {expected_bad_count} bad records for scenario '{test_scenario}'"
        
        # Define the expected schema - note the column names match the actual output from create_enriched_order_details
        expected_schema = StructType([
            StructField("Order_ID", StringType(), True),
            StructField("Order_Date", StringType(), True),
            StructField("Customer_ID", StringType(), True),
            StructField("Customer_Name", StringType(), True),
            StructField("Country", StringType(), True),
            StructField("Category", StringType(), True),
            StructField("Sub_Category", StringType(), True),
            StructField("Profit", DoubleType(), True),
            # Audit fields
            StructField("created_at", TimestampType(), True),
            StructField("modified_at", TimestampType(), True),
            StructField("created_job_id", StringType(), True),
            StructField("created_run_id", StringType(), True),
            StructField("modified_job_id", StringType(), True),
            StructField("modified_run_id", StringType(), True)
        ])
        
        # For negative profit scenarios, we expect no valid records
        if profit_value < 0:
            expected_data = []
        else:
            expected_data = [(
                f"O1_{test_scenario}",  # Order_ID
                "21/8/2016",            # Order_Date (in the input format, not standardized)
                "C1",                   # Customer_ID
                "Alice",                # Customer_Name
                "USA",                  # Country
                "Furniture",            # Category
                "Chairs",               # Sub_Category
                float(profit_value),     # Profit (already rounded to 2 decimals in function)
                # Audit fields (will be set by function)
                None,  # created_at (will be set to current timestamp)
                None,  # modified_at (will be set to current timestamp)
                "manual_job",           # created_job_id
                "manual_run",           # created_run_id
                "manual_job",           # modified_job_id
                "manual_run"            # modified_run_id
            )]
        
        expected_df = spark.createDataFrame(expected_data, schema=expected_schema)
        
        # Define the non-audit columns to compare
        non_audit_columns = [
            "Order_ID", "Order_Date", "Customer_ID", "Customer_Name", 
            "Country", "Category", "Sub_Category", "Profit"
        ]
        
        # Select only the non-audit columns for comparison
        actual_selected = valid_df.select(non_audit_columns)
        expected_selected = expected_df.select(non_audit_columns)
        
        # Sort both DataFrames for consistent comparison
        actual_sorted = actual_selected.orderBy("Order_ID")
        expected_sorted = expected_selected.orderBy("Order_ID")
        
        # Compare DataFrames
        actual_data = actual_sorted.collect()
        expected_data = expected_sorted.collect()
        
        # For debugging
        if actual_data != expected_data:
            print(f"\n=== Data Mismatch for scenario '{test_scenario}' ===")
            print("Expected:", expected_data)
            print("Actual:", actual_data)
            print("\n=== Valid DF Schema ===")
            valid_df.printSchema()
            print("\n=== Valid DF Data ===")
            valid_df.show(truncate=False)
        
        assert actual_data == expected_data, (
            f"Data mismatch for scenario '{test_scenario}'.\n"
            f"Expected: {expected_data}\n"
            f"Actual: {actual_data}"
        )
    
    @pytest.mark.parametrize("date_format,is_valid,expected_date", [
        ("21/8/2016", True, "2016-08-21"),      # Valid DD/M/YYYY format
        ("01/12/2020", True, "2020-12-01"),     # Valid DD/MM/YYYY format
        ("2020-12-31", True, "2020-12-31"),     # Valid YYYY-MM-DD format
        ("31/12/2020", True, "2020-12-31"),     # Valid DD/MM/YYYY format with 2-digit day/month
        ("1/1/2020", True, "2020-01-01"),       # Valid D/M/YYYY format with single digit day/month
        ("invalid_date", False, None),           # Invalid date format
    ])

    # Test case 2 - Parameterized test for date format validation
    def test_date_format_validation(self, spark, date_format: str, is_valid: bool, expected_date: str):
        order_id = f"O_{date_format.replace('/', '_')}"  # Unique order ID based on date format
        orders_data = [(order_id, date_format, "C1", "P1", 100.0)]
        customers_data = [("C1", "Alice", "USA")]
        products_data = [("P1", "Furniture", "Chairs")]
        
        # Create DataFrames with explicit schemas
        orders_df = spark.createDataFrame(orders_data, schema=ORDERS_SCHEMA)
        customers_df = spark.createDataFrame(customers_data, schema=CUSTOMERS_SCHEMA)
        products_df = spark.createDataFrame(products_data, schema=PRODUCTS_SCHEMA)
        
        # Call the function under test with DataFrames directly
        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders=orders_df,
            customers=customers_df,
            products=products_df,
            save=False
        )
        
        if is_valid and expected_date:
            # Define expected schema for the valid output
            expected_schema = StructType([
                StructField("Order ID", StringType()),
                StructField("Order Date", StringType()),
                StructField("Customer ID", StringType()),
                StructField("Customer Name", StringType()),
                StructField("Country", StringType()),
                StructField("Category", StringType()),
                StructField("Sub-Category", StringType()),
                StructField("Profit", DoubleType()),
                StructField("created_at", TimestampType()),
                StructField("modified_at", TimestampType()),
                StructField("created_job_id", StringType()),
                StructField("created_run_id", StringType()),
                StructField("modified_job_id", StringType()),
                StructField("modified_run_id", StringType())
            ])
            
            # Create expected DataFrame with the correct column names (matching the actual output)
            expected_data = [
                (
                    order_id,           # Order ID
                    expected_date,      # Order Date (standardized format)
                    "C1",               # Customer ID
                    "Alice",            # Customer Name
                    "USA",              # Country
                    "Furniture",        # Category
                    "Chairs",           # Sub-Category
                    100.0,              # Profit
                    None, None, None, None, None, None  # Audit fields
                )
            ]
            
            expected_df = spark.createDataFrame(expected_data, schema=expected_schema)
            
            # Verify we have exactly one valid record and no bad records
            assert valid_df.count() == 1, f"Expected 1 valid record for date format: {date_format}"
            assert bad_df.count() == 0, f"Expected 0 bad records for valid date format: {date_format}"
            
            # Select only the columns we want to compare (excluding timestamp fields)
            cols_to_compare = ["Order ID", "Order Date", "Customer ID", "Customer Name", 
                             "Country", "Category", "Sub-Category", "Profit"]
            
            # Compare DataFrames
            actual_selected = valid_df.select(cols_to_compare)
            expected_selected = expected_df.select(cols_to_compare)
            
            # Sort both DataFrames for consistent comparison
            actual_sorted = actual_selected.orderBy("Order ID")
            expected_sorted = expected_selected.orderBy("Order ID")
            
            # Compare the DataFrames
            assert actual_sorted.collect() == expected_sorted.collect(), (
                f"Data mismatch for date format '{date_format}'.\n"
                f"Expected: {expected_sorted.collect()}\n"
                f"Actual: {actual_sorted.collect()}"
            )
            
        else:
            # For invalid dates, we expect at least one bad record
            assert bad_df.count() >= 1, f"Expected at least 1 bad record for invalid date format: {date_format}"
            bad_rows = bad_df.collect()
            assert any(bad_row['Order ID'] == order_id for bad_row in bad_rows), \
                f"Bad record not found for order {order_id} with date format: {date_format}"
    
    # Test Case 4 - Check if profit values are rounded to 2 decimal places
    def test_profit_rounding(self, spark):
        # Test data with various decimal precisions
        orders_data = [
            ("O1", "21/8/2016", "C1", "P1", 100.123456),  # Should round to 100.12
            ("O2", "22/8/2016", "C1", "P2", 50.789012),   # Should round to 50.79
            ("O3", "23/8/2016", "C1", "P3", 25.5),        # Should become 25.50
            ("O4", "24/8/2016", "C1", "P4", 75.0)         # Should become 75.00
        ]
        customers_data = [("C1", "Alice", "USA")]
        products_data = [
            ("P1", "Furniture", "Chairs"),
            ("P2", "Furniture", "Tables"),
            ("P3", "Office Supplies", "Pens"),
            ("P4", "Office Supplies", "Paper")
        ]
        
        # Create DataFrames
        orders_df = spark.createDataFrame(orders_data, schema=ORDERS_SCHEMA)
        customers_df = spark.createDataFrame(customers_data, schema=CUSTOMERS_SCHEMA)
        products_df = spark.createDataFrame(products_data, schema=PRODUCTS_SCHEMA)

        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders=orders_df,
            customers=customers_df,
            products=products_df,
            save=False
        )

        # Define expected schema for the valid output
        expected_schema = StructType([
            StructField("Order ID", StringType()),
            StructField("Order Date", StringType()),
            StructField("Customer ID", StringType()),
            StructField("Customer Name", StringType()),
            StructField("Country", StringType()),
            StructField("Category", StringType()),
            StructField("Sub-Category", StringType()),
            StructField("Profit", DoubleType()),
            StructField("created_at", TimestampType()),
            StructField("modified_at", TimestampType()),
            StructField("created_job_id", StringType()),
            StructField("created_run_id", StringType()),
            StructField("modified_job_id", StringType()),
            StructField("modified_run_id", StringType())
        ])
        
        # Create expected DataFrame with the correct column names and rounded profit values
        expected_valid_data = [
            (
                "O1",               # Order ID
                "2016-08-21",       # Order Date (standardized format)
                "C1",               # Customer ID
                "Alice",            # Customer Name
                "USA",              # Country
                "Furniture",        # Category
                "Chairs",           # Sub-Category
                100.12,             # Rounded Profit
                None, None, None, None, None, None  # Audit fields
            ),
            (
                "O2",               # Order ID
                "2016-08-22",       # Order Date (standardized format)
                "C1",               # Customer ID
                "Alice",            # Customer Name
                "USA",              # Country
                "Furniture",        # Category
                "Tables",           # Sub-Category
                50.79,              # Rounded Profit
                None, None, None, None, None, None  # Audit fields
            ),
            (
                "O3",               # Order ID
                "2016-08-23",       # Order Date (standardized format)
                "C1",               # Customer ID
                "Alice",            # Customer Name
                "USA",              # Country
                "Office Supplies",  # Category
                "Pens",             # Sub-Category
                25.5,               # Rounded Profit
                None, None, None, None, None, None  # Audit fields
            ),
            (
                "O4",               # Order ID
                "2016-08-24",       # Order Date (standardized format)
                "C1",               # Customer ID
                "Alice",            # Customer Name
                "USA",              # Country
                "Office Supplies",  # Category
                "Paper",            # Sub-Category
                75.0,               # Rounded Profit
                None, None, None, None, None, None  # Audit fields
            )
        ]
        
        expected_valid_df = spark.createDataFrame(expected_valid_data, schema=expected_schema)
        
        # Select only the columns we want to compare (excluding timestamp fields)
        cols_to_compare = ["Order ID", "Order Date", "Customer ID", "Customer Name", 
                          "Country", "Category", "Sub-Category", "Profit"]
        
        # Compare valid DataFrames
        actual_valid_selected = valid_df.select(cols_to_compare)
        expected_valid_selected = expected_valid_df.select(cols_to_compare)
        
        # Sort both DataFrames for consistent comparison
        actual_sorted = actual_valid_selected.orderBy("Order ID")
        expected_sorted = expected_valid_selected.orderBy("Order ID")
        
        # Compare the DataFrames
        assert actual_sorted.collect() == expected_sorted.collect(), (
            "Data mismatch in profit rounding test.\n"
            f"Expected: {expected_sorted.collect()}\n"
            f"Actual: {actual_sorted.collect()}"
        )
        
        # Verify no bad records for this test case
        assert bad_df.count() == 0, f"Expected 0 bad records, got {bad_df.count()}"
