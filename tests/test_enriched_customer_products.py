import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from pyspark.sql.utils import AnalysisException
import sys
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Dynamically add the repo root to sys.path
repo_root = "/Workspace/Users/pare.garima.18@gmail.com/PEI_Ecommerce/src"
sys.path.append(repo_root)

from src.enriched_customers_products import create_enriched_customers, create_enriched_products

@pytest.fixture(scope="session")
def spark():
    """
    Create a new Spark session for pytest when running inside Databricks via shell.
    """
    return (
        SparkSession.builder
        .appName("pytest-enriched")
        .getOrCreate()
    )

class TestEnrichedCustomersExtended:
    """Test cases for create_enriched_customers function"""
    
    # Define the expected schema for customer data that will be used across tests
    CUSTOMER_SCHEMA = StructType([
        StructField("Customer_ID", StringType(), True),
        StructField("Customer_Name", StringType(), False),
        StructField("Email", StringType(), False),
        StructField("Phone", StringType(), True),
        StructField("Address", StringType(), True),
        StructField("Segment", StringType(), True),
        StructField("Country", StringType(), True),
        StructField("City", StringType(), True),
        StructField("State", StringType(), True),
        StructField("Postal_Code", StringType(), True),
        StructField("Region", StringType(), True)
    ])
    
    # Define the input schema for customer data
    INPUT_CUSTOMER_SCHEMA = StructType([
        StructField("Customer ID", StringType(), True),
        StructField("Customer Name", StringType(), False),
        StructField("email", StringType(), False),
        StructField("phone", StringType(), True),
        StructField("address", StringType(), True),
        StructField("Segment", StringType(), True),
        StructField("Country", StringType(), True),
        StructField("City", StringType(), True),
        StructField("State", StringType(), True),
        StructField("Postal Code", StringType(), True),
        StructField("Region", StringType(), True)
    ])

    def test_customers_missing_columns_error(self, spark):
        # Test Case 1 - Source table with missing columns 
        incomplete_data = [
            Row(**{"Customer ID": "C1", "Customer Name": "Alice"})  # Missing many required columns
        ]
        
        incomplete_df = spark.createDataFrame(incomplete_data)
        incomplete_df.createOrReplaceTempView("incomplete_customers")

        with pytest.raises(RuntimeError) as e:
            create_enriched_customers(
                spark,
                customers="incomplete_customers",
                save=False
            )
        assert "Customers table missing" in str(e.value)


    def test_customers_all_valid_records(self, spark):
        # Test Case 2 - Customer Source table with all valid records.
        # Check if the records are processed correctly and renaming of columns is done correctly
        data = [
            ("C1", "Alice", "alice@test.com", "123", "Addr1", "Consumer", "USA", "NY", "NY", "10001", "East"),
            ("C2", "Bob", "bob@test.com", "456", "Addr2", "Corporate", "USA", "SF", "CA", "94016", "West"),
        ]
        
        # Create input DataFrame using the class-level input schema
        input_df = spark.createDataFrame(data, schema=self.INPUT_CUSTOMER_SCHEMA)
        input_df.createOrReplaceTempView("all_valid_customers")

        # Call the function under test
        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="all_valid_customers",
            save=False
        )
        
        # Create expected DataFrame using the class-level output schema
        expected_data = [
            ("C1", "Alice", "alice@test.com", "123", "Addr1", "Consumer", "USA", "NY", "NY", "10001", "East"),
            ("C2", "Bob", "bob@test.com", "456", "Addr2", "Corporate", "USA", "SF", "CA", "94016", "West")
        ]
        expected_df = spark.createDataFrame(expected_data, schema=self.CUSTOMER_SCHEMA)
        
        # Verify counts
        assert valid_df.count() == 2, f"Expected 2 valid records, got {valid_df.count()}"
        assert bad_df.count() == 0, f"Expected 0 bad records, got {bad_df.count()}"
        
        # Select only the expected columns from the actual result (excluding audit columns)
        actual_selected = valid_df.select(self.CUSTOMER_SCHEMA.fieldNames())
        
        # Sort both DataFrames by Customer_ID for consistent comparison
        actual_sorted = actual_selected.orderBy("Customer_ID").collect()
        expected_sorted = expected_df.orderBy("Customer_ID").collect()
        
        # Compare the sorted DataFrames
        assert actual_sorted == expected_sorted, "The actual data does not match the expected data"


    def test_cust_records_with_nulls(self, spark):
        # Test Case 3 - Customer Source table with some null values
        # Check if records with null values in non-required fields are handled correctly
        
        # Input data with null values in non-required fields
        input_data = [
            ("C1", "Alice", "alice@test.com", "123", "Addr1", "Consumer", "USA", "NY", "NY", "10001", "East"),
            ("C2", "Bob", "bob@test.com", "456", None, "Corporate", "USA", "SF", "CA", "94016", "West")
        ]
        
        # Create input DataFrame using the class-level input schema
        input_df = spark.createDataFrame(input_data, schema=self.INPUT_CUSTOMER_SCHEMA)
        input_df.createOrReplaceTempView("customers_with_nulls")

        # Call the function under test
        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="customers_with_nulls",
            save=False
        )
        
        # Expected output data (should match input since nulls are in allowed fields)
        expected_data = [
            ("C1", "Alice", "alice@test.com", "123", "Addr1", "Consumer", "USA", "NY", "NY", "10001", "East"),
            ("C2", "Bob", "bob@test.com", "456", None, "Corporate", "USA", "SF", "CA", "94016", "West")
        ]
        
        # Create expected DataFrame using the class-level output schema
        expected_df = spark.createDataFrame(expected_data, schema=self.CUSTOMER_SCHEMA)

        # Verify counts
        assert valid_df.count() == 2, f"Expected 2 valid records, got {valid_df.count()}"
        assert bad_df.count() == 0, f"Expected 0 bad records, got {bad_df.count()}"
        
        # Select only the expected columns from the actual result (excluding audit columns)
        actual_selected = valid_df.select(self.CUSTOMER_SCHEMA.fieldNames())
        
        # Sort both DataFrames by Customer_ID for consistent comparison
        actual_sorted = actual_selected.orderBy("Customer_ID").collect()
        expected_sorted = expected_df.orderBy("Customer_ID").collect()
        
        # Compare the sorted DataFrames
        assert actual_sorted == expected_sorted, "The valid records do not match the expected data"

    def test_customers_all_invalid_records(self, spark):
        #Test Case 3 - Customer Source table with all invalid records.Check if bad records are handled correctly
        data = [
            Row(**{"Customer ID": None, "Customer Name": "Alice", "email": "alice@test.com", 
                "phone": "123", "address": "Addr1", "Segment": "Consumer", "Country": "USA", 
                "City": "NY", "State": "NY", "Postal Code": "10001", "Region": "East"}),
            Row(**{"Customer ID": None, "Customer Name": "Bob", "email": "bob@test.com", 
                "phone": "456", "address": "Addr2", "Segment": "Corporate", "Country": "USA", 
                "City": "SF", "State": "CA", "Postal Code": "94016", "Region": "West"}),
        ]
        
        # Use the class-level input schema for consistency
        df = spark.createDataFrame(data, schema=self.INPUT_CUSTOMER_SCHEMA)
        df.createOrReplaceTempView("all_invalid_customers")

        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="all_invalid_customers",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 2

    def test_customers_empty_dataframe(self, spark):
        # Test Case 4 - Customer Source table with empty dataframe.
        # Check if the records are processed correctly
        empty_df = spark.createDataFrame([], schema=self.INPUT_CUSTOMER_SCHEMA)

        empty_df.createOrReplaceTempView("empty_customers")

        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="empty_customers",
            save=False
        ) 

        assert valid_df.count() == 0
        assert bad_df.count() == 0


class TestEnrichedProducts:
    # Define schemas for products
    INPUT_PRODUCT_SCHEMA = StructType([
        StructField("Product ID", StringType(), True),  # Made nullable for test data
        StructField("Category", StringType(), True),
        StructField("Sub-Category", StringType(), True),
        StructField("Product Name", StringType(), True),  # Made nullable for test data
        StructField("State", StringType(), True),
        StructField("Price per product", DoubleType(), True)
    ])
    
    # Schema for expected output (without audit columns for comparison)
    PRODUCT_SCHEMA = StructType([
        StructField("Product_ID", StringType(), True),
        StructField("Category", StringType(), True),
        StructField("Sub_Category", StringType(), True),
        StructField("Product_Name", StringType(), True),
        StructField("State", StringType(), True),
        StructField("Price_Per_Product", DoubleType(), True)
    ])
    
    # Full schema including audit columns
    FULL_PRODUCT_SCHEMA = StructType([
        StructField("Product_ID", StringType(), True),
        StructField("Category", StringType(), True),
        StructField("Sub_Category", StringType(), True),
        StructField("Product_Name", StringType(), True),
        StructField("State", StringType(), True),
        StructField("Price_Per_Product", DoubleType(), True),
        StructField("Created_Timestamp", TimestampType(), True),
        StructField("Updated_Timestamp", TimestampType(), True)
    ])

    def compare_dataframes(self, actual_df, expected_data, schema=None, spark=None):
        """Helper method to compare DataFrames, skipping audit columns."""
        if schema is None:
            schema = self.PRODUCT_SCHEMA
            
        # Create expected DataFrame
        if spark is None:
            spark = actual_df.sparkSession
        expected_df = spark.createDataFrame(expected_data, schema=self.PRODUCT_SCHEMA)
        
        # Get non-audit column names for comparison
        non_audit_columns = [f for f in schema.fieldNames() 
                           if f not in ['Created_Timestamp', 'Updated_Timestamp']]
        
        # Select only the non-audit columns from both DataFrames
        actual_selected = actual_df.select(non_audit_columns)
        expected_selected = expected_df.select(non_audit_columns)
        
        # Sort both DataFrames by Product_ID for consistent comparison
        actual_sorted = actual_selected.orderBy("Product_ID")
        expected_sorted = expected_selected.orderBy("Product_ID")
        

    def test_products_missing_columns_error(self, spark):
        # Test Case 1 - Source table with missing columns      
        incomplete_data = [
            Row(**{"Product ID": "P1", "Category": "Tech"})  # Missing required columns
        ]
        
        incomplete_df = spark.createDataFrame(incomplete_data)
        incomplete_df.createOrReplaceTempView("incomplete_products")

        with pytest.raises(RuntimeError) as e:
            create_enriched_products(
                spark,
                products="incomplete_products",
                save=False
            )
        assert "Products table missing" in str(e.value)

    def test_products_all_valid_records(self, spark):
        # Test Case 2 - Product Source table with all valid records.
        # Check if the records are processed correctly and renaming of columns is done correctly
        data = [
            ("P1", "Furniture", "Chairs", "Office Chair", "CA", 200.0),
            ("P2", "Tech", "Phones", "Smartphone", "NY", 500.0),
        ]
        
        # Create input DataFrame using the class-level input schema
        input_df = spark.createDataFrame(data, schema=self.INPUT_PRODUCT_SCHEMA)
        input_df.createOrReplaceTempView("all_valid_products")

        # Call the function under test
        valid_df, bad_df = create_enriched_products(
            spark,
            products="all_valid_products",
            save=False
        )
        
        # Create expected DataFrame using the class-level output schema
        expected_data = [
            ("P1", "Furniture", "Chairs", "Office Chair", "CA", 200.0),
            ("P2", "Tech", "Phones", "Smartphone", "NY", 500.0)
        ]
        
        # Sort both DataFrames by Product_ID for consistent comparison

        expected_df = spark.createDataFrame(expected_data, schema=self.PRODUCT_SCHEMA)

        # Select only the expected columns from the actual result (excluding audit columns)
        actual_selected = valid_df.select(self.PRODUCT_SCHEMA.fieldNames())

        # Sort both DataFrames by Product_ID for consistent comparison
        actual_sorted = actual_selected.orderBy("Product_ID")
        expected_sorted = expected_df.orderBy("Product_ID")
        

        assert actual_sorted == expected_sorted, "The valid records do not match the expected data"
        # Verify counts
        assert valid_df.count() == 2, f"Expected 2 valid records, got {valid_df.count()}"
        assert bad_df.count() == 0, f"Expected 0 bad records, got {bad_df.count()}"



    def test_products_all_invalid_records(self, spark):
        # Test Case 3 - Product Source table with all invalid records.
        # Check if bad records are handled correctly
        
        # Create test data with invalid records
        data = [
            (None, None, None, "Invalid Product", None, None),  # Missing required fields
            ("", "", "", "", "", -100.0),  # Empty strings and negative price
        ]
        
        # Create input DataFrame using the class-level input schema
        input_df = spark.createDataFrame(data, schema=self.INPUT_PRODUCT_SCHEMA)
        input_df.createOrReplaceTempView("all_invalid_products")

        # Call the function under test
        valid_df, bad_df = create_enriched_products(
            spark,
            products="all_invalid_products",
            save=False
        )

        # Verify counts
        assert valid_df.count() == 0, "Expected no valid records"
        assert bad_df.count() == 2, f"Expected 2 bad records, got {bad_df.count()}"

        expected_df = spark.createDataFrame(expected_data, schema=self.PRODUCT_SCHEMA)

        # Select only the expected columns from the actual result (excluding audit columns)
        actual_selected = valid_df.select(self.PRODUCT_SCHEMA.fieldNames())

        # Sort both DataFrames by Product_ID for consistent comparison
        actual_sorted = actual_selected.orderBy("Product_ID")
        expected_sorted = expected_df.orderBy("Product_ID")
       
        assert actual_sorted == expected_sorted, "The valid records do not match the expected data"

    def test_products_boundary_prices(self, spark):
        # Test Case 4 - Check if boundary price values are handled correctly
        data = [
            ("P1", "Tech", "Phones", "Phone", "NY", 500.0),     # Valid price
            ("P2", "Tech", "Phones", "Phone", "NY", 999999.99), # Valid large price
            ("P3", "Tech", "Phones", "Phone", "NY", 0.0),        # Zero price (invalid)
            ("P4", "Tech", "Phones", "Phone", "NY", -20.0)       # Negative price (invalid)
        ]
        
        # Create input DataFrame using the class-level input schema
        input_df = spark.createDataFrame(data, schema=self.INPUT_PRODUCT_SCHEMA)
        input_df.createOrReplaceTempView("boundary_products")
        
        # Call the function under test
        valid_df, bad_df = create_enriched_products(
            spark,
            products="boundary_products",
            save=False
        )

        # Verify counts
        assert valid_df.count() == 2, f"Expected 2 valid records, got {valid_df.count()}"
        assert bad_df.count() == 2, f"Expected 2 bad records, got {bad_df.count()}"

        # Create expected data for valid records (without timestamps)
        expected_valid_data = [
            ("P1", "Tech", "Phones", "Phone", "NY", 500.0),    # Valid price
            ("P2", "Tech", "Phones", "Phone", "NY", 999999.99) # Valid large price
        ]
        
        expected_df = spark.createDataFrame(expected_data, schema=self.PRODUCT_SCHEMA)

        # Select only the expected columns from the actual result (excluding audit columns)
        actual_selected = valid_df.select(self.PRODUCT_SCHEMA.fieldNames())

        # Sort both DataFrames by Product_ID for consistent comparison
        actual_sorted = actual_selected.orderBy("Product_ID")
        expected_sorted = expected_df.orderBy("Product_ID")
        
        assert actual_sorted == expected_sorted, "The valid records do not match the expected data"

    def test_products_empty_dataframe(self, spark):
        # Test Case 4 - Check if job runs with empty products dataframe
        
        # Create empty DataFrame using the class-level input schema
        empty_df = spark.createDataFrame([], schema=self.INPUT_PRODUCT_SCHEMA)
        empty_df.createOrReplaceTempView("empty_products")

        # Call the function under test
        valid_df, bad_df = create_enriched_products(
            spark,
            products="empty_products",
            save=False
        )

        # Verify counts
        assert valid_df.count() == 0, "Expected no valid records for empty input"
        assert bad_df.count() == 0, "Expected no bad records for empty input"
        
    def test_mixed_valid_invalid_products(self, spark):
        # Test Case 5 - Check if combination of valid and bad records are handled correctly
        data = [
            # Valid product 1
            ("P1", "Tech", "Phones", "Phone", "NY", 1000.0),
            # Valid product 2
            ("P2", "Furniture", "Chairs", "Office Chair", "CA", 200.0),
            # Invalid product (missing required fields)
            (None, None, None, None, None, None),
            # Another invalid product (negative price)
            ("P3", "Tech", "Laptops", "Laptop", "TX", -500.0),
            # Another invalid product (zero price)
            ("P4", "Office Supplies", "Paper", "A4 Paper", "IL", 0.0)
        ]
        
        # Create input DataFrame using the class-level input schema
        input_df = spark.createDataFrame(data, schema=self.INPUT_PRODUCT_SCHEMA)
        input_df.createOrReplaceTempView("mixed_products")

        # Call the function under test
        valid_df, bad_df = create_enriched_products(
            spark,
            products="mixed_products",
            save=False
        )

        # Verify counts
        assert valid_df.count() == 2, f"Expected 2 valid records, got {valid_df.count()}"
        assert bad_df.count() == 3, f"Expected 3 bad records, got {bad_df.count()}"
        
        # Create expected data for valid records (without timestamps)
        expected_valid_data = [
            ("P1", "Tech", "Phones", "Phone", "NY", 1000.0),
            ("P2", "Furniture", "Chairs", "Office Chair", "CA", 200.0)
        ]
        
        # Compare the DataFrames using the helper method
        self.compare_dataframes(valid_df, expected_valid_data, spark=spark)
        
        # Verify bad records contain the expected invalid data (checking counts only)
        bad_prices = [row.Price_Per_Product for row in bad_df.collect() 
                     if row.Price_Per_Product is not None]
        assert -500.0 in bad_prices, "Expected record with negative price in bad records"
        assert 0.0 in bad_prices, "Expected record with zero price in bad records"
