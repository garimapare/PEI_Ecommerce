
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import Row
from src.enriched_customers_products import create_enriched_customers, create_enriched_products

# -----------------------------
# Spark session fixture
# -----------------------------
@pytest.fixture(scope="session")
def spark():
    """
    Return the active Spark session in Databricks.
    Cannot create a local Spark session when using Spark Connect.
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        pytest.skip("No active Spark session available.")
    return spark

# -----------------------------
# Test enriched customers
# -----------------------------
def test_valid_customers(spark):
    # now 'spark' is guaranteed to be the active Databricks session
    data = [
        ("C1", "Alice", "alice@mail.com", "123", "Addr1", "Consumer", "USA", "NY", "NY", "10001", "East"),
        ("C2", None, "bob@mail.com", "456", "Addr2", "Corporate", "USA", "SF", "CA", "94016", "West"),
    ]
    cols = ["Customer ID", "Customer Name", "Email", "Phone", "Address",
            "Segment", "Country", "City", "State", "Postal Code", "Region"]
    df = spark.createDataFrame(data, cols)
    
    valid_df, bad_df = create_enriched_customers(spark, customers=df, save=False)
    
    assert valid_df.count() == 1
    assert bad_df.count() == 1
    valid_df.show()

    # Expected DataFrames
    expected_valid = spark.createDataFrame([
        ("C1", "Alice", "alice@mail.com", "123", "Addr1", "Consumer", "USA", "NY", "NY", "10001", "East")
    ], ["Customer_ID", "Customer_Name", "Email", "Phone", "Address", 
        "Segment", "Country", "City", "State", "Postal_Code", "Region"])

    expected_bad = spark.createDataFrame([
        ("C2", None, "bob@mail.com", "456", "Addr2", "Corporate", "USA", "SF", "CA", "94016", "West"),
        (None, "Charlie", "charlie@mail.com", "789", "Addr3", "Consumer", "USA", "LA", "CA", "90001", "West")
    ], ["Customer_ID", "Customer_Name", "Email", "Phone", "Address", 
        "Segment", "Country", "City", "State", "Postal_Code", "Region"])

    # Compare row sets
    assert set([tuple(r) for r in valid_df.select(*expected_valid.columns).collect()]) == set([tuple(r) for r in expected_valid.collect()])
    assert set([tuple(r) for r in bad_df.select(*expected_bad.columns).collect()]) == set([tuple(r) for r in expected_bad.collect()])

# -----------------------------
# Test enriched products
# -----------------------------
def test_valid_products(spark):
    # Input DataFrame
    data = [
        ("P1", "Furniture", "Chairs", "Office Chair", "CA", 200.0),
        ("P2", "Tech", "Phones", "iPhone", "NY", -10.0),
        ("P3", "Furniture", "Tables", "Dining Table", "TX", 0.0)
    ]
    cols = ["Product ID", "Category", "Sub-Category", "Product Name", "State", "Price per product"]
    input_df = spark.createDataFrame(data, cols)

    # Call enrichment function
    valid_df, bad_df = create_enriched_products(
        spark,
        products=input_df,
        save=False
    )

    # Expected DataFrames
    expected_valid = spark.createDataFrame([
        ("P1", "Furniture", "Chairs", "Office Chair", "CA", 200.0)
    ], ["Product_ID", "Category", "Sub_Category", "Product_Name", "State", "Price_Per_Product"])

    expected_bad = spark.createDataFrame([
        ("P2", "Tech", "Phones", "iPhone", "NY", -10.0),
        ("P3", "Furniture", "Tables", "Dining Table", "TX", 0.0)
    ], ["Product_ID", "Category", "Sub_Category", "Product_Name", "State", "Price_Per_Product"])

    # Compare row sets
    assert set([tuple(r) for r in valid_df.select(*expected_valid.columns).collect()]) == set([tuple(r) for r in expected_valid.collect()])
    assert set([tuple(r) for r in bad_df.select(*expected_bad.columns).collect()]) == set([tuple(r) for r in expected_bad.collect()])
