# import pytest
# from pyspark.sql import SparkSession
# from pyspark.sql import Row
# from pyspark.sql.utils import AnalysisException
# import sys
# import os
# import os
# os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# # Dynamically add the repo root to sys.path
# repo_root = "/Workspace/Users/pare.garima.18@gmail.com/PEI_Ecommerce/src"
# sys.path.append(repo_root)

# from src.enriched_customers_products import create_enriched_customers, create_enriched_products

# @pytest.fixture(scope="session")
# def spark():
#     """
#     Create a new Spark session for pytest when running inside Databricks via shell.
#     """
#     return (
#         SparkSession.builder
#         .appName("pytest-enriched")
#         .getOrCreate()
#     )

import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
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
        #Test Case 2 - Customer Source table with all valid records.Check if the records are processed correctly and renaming of columns is done correctly
        data = [
            Row(**{"Customer ID": "C1", "Customer Name": "Alice", "email": "alice@test.com", 
                "phone": "123", "address": "Addr1", "Segment": "Consumer", "Country": "USA", 
                "City": "NY", "State": "NY", "Postal Code": "10001", "Region": "East"}),
            Row(**{"Customer ID": "C2", "Customer Name": "Bob", "email": "bob@test.com", 
                "phone": "456", "address": "Addr2", "Segment": "Corporate", "Country": "USA", 
                "City": "SF", "State": "CA", "Postal Code": "94016", "Region": "West"}),
        ]
        
        df = spark.createDataFrame(data)
        df.createOrReplaceTempView("all_valid_customers")

        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="all_valid_customers",
            save=False
        )

        assert valid_df.count() == 2
        assert bad_df.count() == 0

        # Check column renaming
        columns = valid_df.columns
        assert "Customer_ID" in columns
        assert "Customer_Name" in columns
        assert "Postal_Code" in columns

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
        
        # Define explicit schema to handle null Customer ID values
        schema = StructType([
            StructField("Customer ID", StringType(), True),
            StructField("Customer Name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("phone", StringType(), True),
            StructField("address", StringType(), True),
            StructField("Segment", StringType(), True),
            StructField("Country", StringType(), True),
            StructField("City", StringType(), True),
            StructField("State", StringType(), True),
            StructField("Postal Code", StringType(), True),
            StructField("Region", StringType(), True)
        ])
        
        df = spark.createDataFrame(data, schema)
        df.createOrReplaceTempView("all_invalid_customers")

        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="all_invalid_customers",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 2

    def test_customers_empty_dataframe(self, spark):
        #Test Case 4 - Customer Source table with empty dataframe.Check if the records are processed correctly
        empty_df = spark.createDataFrame([], 
            StructType([
                StructField("Customer ID", StringType(), True),
                StructField("Customer Name", StringType(), True),
                StructField("email", StringType(), True),
                StructField("phone", StringType(), True),
                StructField("address", StringType(), True),
                StructField("Segment", StringType(), True),
                StructField("Country", StringType(), True),
                StructField("City", StringType(), True),
                StructField("State", StringType(), True),
                StructField("Postal Code", StringType(), True),
                StructField("Region", StringType(), True)
            ]))

        empty_df.createOrReplaceTempView("empty_customers")

        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="empty_customers",
            save=False
        ) 

        assert valid_df.count() == 0
        assert bad_df.count() == 0


class TestEnrichedProductsExtended:

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
        #Test Case 2 - Customer Source table with all valid records.Check if the records are processed correctly and renaming of columns is done correctly
        data = [
            Row(**{"Product ID": "P1", "Category": "Furniture", "Sub-Category": "Chairs", 
                "Product Name": "Office Chair", "State": "CA", "Price per product": 200.0}),
            Row(**{"Product ID": "P2", "Category": "Tech", "Sub-Category": "Phones", 
                "Product Name": "Smartphone", "State": "NY", "Price per product": 500.0}),
        ]
        
        df = spark.createDataFrame(data)
        df.createOrReplaceTempView("all_valid_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="all_valid_products",
            save=False
        )

        assert valid_df.count() == 2
        assert bad_df.count() == 0

        # Check column renaming
        columns = valid_df.columns
        assert "Product_ID" in columns
        assert "Category" in columns
        assert "Sub_Category" in columns
        assert "Product_Name" in columns
        assert "Price_Per_Product" in columns
        assert "Created_Timestamp" in columns
        assert "Updated_Timestamp" in columns

    def test_products_all_invalid_records(self, spark):
        #Test Case 3 - Customer Source table with all invalid records.Check if bad records are handled correctly
        data = [
                # Negative price
            Row(**{"Product ID": "P1", "Category": "Tech", "Sub-Category": "Phones", 
                "Product Name": "Phone", "State": "NY", "Price per product": -100.0}),  
                # Zero price
            Row(**{"Product ID": "P2", "Category": "Furniture", "Sub-Category": "Tables", 
                "Product Name": "Table", "State": "TX", "Price per product": 0.0}),     
                # Null Product_ID
            Row(**{"Product ID": None, "Category": "Tech", "Sub-Category": "Laptops", 
                "Product Name": "Laptop", "State": "CA", "Price per product": 1000.0}), 
        ]
        
        df = spark.createDataFrame(data)
        df.createOrReplaceTempView("all_invalid_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="all_invalid_products",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 3

    def test_products_boundary_prices(self, spark):
        #Test Case 4 -  Products with negative positive,zero and valid price values/Check if negative values are handled as bad record and all valid values are processed correctly"""
        data = [
            # product with positive
            Row(**{"Product ID": "P1", "Category": "Tech", "Sub-Category": "Phones", 
                "Product Name": "Phone", "State": "NY", "Price per product": 500.0}),  
            # product with Very large with decimal  
            Row(**{"Product ID": "P2", "Category": "Tech", "Sub-Category": "Phones", 
                "Product Name": "Phone", "State": "NY", "Price per product": 999999.99}), 
            # product with Exactly zero price
            Row(**{"Product ID": "P3", "Category": "Tech", "Sub-Category": "Phones", 
                "Product Name": "Phone", "State": "NY", "Price per product": 0.0}),     
            # product with negative price
            Row(**{"Product ID": "P4", "Category": "Tech", "Sub-Category": "Phones", 
                "Product Name": "Phone", "State": "NY", "Price per product": -20.0})    
        ]
        
        df = spark.createDataFrame(data)
        df.createOrReplaceTempView("boundary_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="boundary_products",
            save=False
        )

        assert valid_df.count() == 2  # P1 and P2 (positive prices)
        assert bad_df.count() == 2   # P3 and P4 (zero and negative)


        valid_prices = [row["Price_Per_Product"] for row in valid_df.collect()]
        assert 500 in valid_prices
        assert 999999.99 in valid_prices

    def test_products_empty_dataframe(self, spark):
        #Test Case 4 - Check if job runs with empty products dataframe
        empty_df = spark.createDataFrame([], 
            StructType([
                StructField("Product ID", StringType(), True),
                StructField("Category", StringType(), True),
                StructField("Sub-Category", StringType(), True),
                StructField("Product Name", StringType(), True),
                StructField("State", StringType(), True),
                StructField("Price per product", DoubleType(), True)
            ]))

        empty_df.createOrReplaceTempView("empty_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="empty_products",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 0

    def test_mixed_valid_invalid_products(self, spark):
        #Test Case 5 - Check if cobination of valid and bad records are handled correctly and the records are not skipped

        data = [
            # Valid records
            Row(**{"Product ID": "P1", "Category": "Tech", "Sub-Category": "Phones", 
                "Product Name": "iPhone", "State": "CA", "Price per product": 999.0}),
            Row(**{"Product ID": "P2", "Category": "Furniture", "Sub-Category": "Chairs", 
                "Product Name": "Office Chair", "State": "NY", "Price per product": 299.0}),
            
            # Invalid records
            Row(**{"Product ID": "P3", "Category": "Tech", "Sub-Category": "Laptops", 
                "Product Name": "Laptop", "State": "TX", "Price per product": -500.0}),   # Negative price
            Row(**{"Product ID": None, "Category": "Office", "Sub-Category": "Supplies", 
                "Product Name": "Pen", "State": "FL", "Price per product": 5.0}),        # Null Product_ID
            Row(**{"Product ID": "P5", "Category": "Sports", "Sub-Category": "Equipment", 
                "Product Name": "Ball", "State": "WA", "Price per product": 0.0}),       # Zero price
        ]
        
        df = spark.createDataFrame(data)
        df.createOrReplaceTempView("mixed_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="mixed_products",
            save=False
        )

        assert valid_df.count() == 2  # P1 and P2
        assert bad_df.count() == 3   # P3, null ID record, and P5

        # Validate valid products
        valid_ids = [row["Product_ID"] for row in valid_df.collect()]
        assert "P1" in valid_ids
        assert "P2" in valid_ids

        # Validate invalid products
        bad_records = bad_df.collect()
        bad_prices = [row["Price_Per_Product"] for row in bad_records if row["Price_Per_Product"] is not None]
        assert -500.0 in bad_prices
        assert 0.0 in bad_prices
