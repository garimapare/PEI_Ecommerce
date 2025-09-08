import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.utils import AnalysisException
import sys
import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.enriched_customers_products import create_enriched_customers, create_enriched_products


@pytest.fixture(scope="session")
def spark():
    """Provide a Spark session for tests"""
    return SparkSession.builder \
        .appName("pytest-customers-products-extended") \
        .master("local[2]") \
        .getOrCreate()


class TestEnrichedCustomersExtended:
    """Extended test cases for enriched customers functionality"""

    def test_customers_missing_table_error(self, spark):
        """Test error handling when customers table doesn't exist"""
        with pytest.raises(RuntimeError) as e:
            create_enriched_customers(
                spark,
                customers="non_existent.customers",
                save=False
            )
        assert "Source table not found" in str(e.value)

    def test_customers_missing_columns_error(self, spark):
        """Test error handling when required columns are missing"""
        # Create table with missing columns
        incomplete_data = [
            Row(Customer_ID="C1", Customer_Name="Alice")  # Missing many required columns
        ]
        
        incomplete_df = spark.createDataFrame(incomplete_data)
        incomplete_df.write.format("delta").mode("overwrite").saveAsTable("test.incomplete_customers")

        with pytest.raises(RuntimeError) as e:
            create_enriched_customers(
                spark,
                customers="test.incomplete_customers",
                save=False
            )
        assert "Customers table missing" in str(e.value)

    def test_customers_all_valid_records(self, spark):
        """Test scenario where all customer records are valid"""
        data = [
            Row(Customer_ID="C1", Customer_Name="Alice", Email="alice@test.com", 
                Phone="123", Address="Addr1", Segment="Consumer", Country="USA", 
                City="NY", State="NY", Postal_Code="10001", Region="East"),
            Row(Customer_ID="C2", Customer_Name="Bob", Email="bob@test.com", 
                Phone="456", Address="Addr2", Segment="Corporate", Country="USA", 
                City="SF", State="CA", Postal_Code="94016", Region="West"),
        ]
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.all_valid_customers")

        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="test.all_valid_customers",
            save=False
        )

        assert valid_df.count() == 2
        assert bad_df.count() == 0

        # Check column renaming
        columns = valid_df.columns
        assert "Customer_ID" in columns
        assert "Customer_Name" in columns
        assert "Postal_Code" in columns
        assert "Created_Timestamp" in columns
        assert "Updated_Timestamp" in columns

    def test_customers_all_invalid_records(self, spark):
        """Test scenario where all customer records are invalid (null Customer_ID)"""
        data = [
            Row(Customer_ID=None, Customer_Name="Alice", Email="alice@test.com", 
                Phone="123", Address="Addr1", Segment="Consumer", Country="USA", 
                City="NY", State="NY", Postal_Code="10001", Region="East"),
            Row(Customer_ID=None, Customer_Name="Bob", Email="bob@test.com", 
                Phone="456", Address="Addr2", Segment="Corporate", Country="USA", 
                City="SF", State="CA", Postal_Code="94016", Region="West"),
        ]
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.all_invalid_customers")

        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="test.all_invalid_customers",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 2

    def test_customers_empty_dataframe(self, spark):
        """Test behavior with empty customers DataFrame"""
        empty_df = spark.createDataFrame([], 
            StructType([
                StructField("Customer ID", StringType(), True),
                StructField("Customer Name", StringType(), True),
                StructField("Email", StringType(), True),
                StructField("Phone", StringType(), True),
                StructField("Address", StringType(), True),
                StructField("Segment", StringType(), True),
                StructField("Country", StringType(), True),
                StructField("City", StringType(), True),
                StructField("State", StringType(), True),
                StructField("Postal Code", StringType(), True),
                StructField("Region", StringType(), True)
            ]))

        empty_df.write.format("delta").mode("overwrite").saveAsTable("test.empty_customers")

        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="test.empty_customers",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 0

    def test_customers_special_characters(self, spark):
        """Test customers with special characters in data"""
        data = [
            Row(Customer_ID="C'1", Customer_Name="Alice & Bob", Email="alice+bob@test.com", 
                Phone="123-456-7890", Address="123 Main St., Apt #5", Segment="Consumer", 
                Country="Côte d'Ivoire", City="New York", State="NY", 
                Postal_Code="10001-1234", Region="North-East"),
        ]
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.special_customers")

        valid_df, bad_df = create_enriched_customers(
            spark,
            customers="test.special_customers",
            save=False
        )

        assert valid_df.count() == 1
        assert bad_df.count() == 0

        row = valid_df.collect()[0]
        assert row["Customer_Name"] == "Alice & Bob"
        assert row["Country"] == "Côte d'Ivoire"
        assert row["Email"] == "alice+bob@test.com"


class TestEnrichedProductsExtended:
    """Extended test cases for enriched products functionality"""

    def test_products_missing_table_error(self, spark):
        """Test error handling when products table doesn't exist"""
        with pytest.raises(RuntimeError) as e:
            create_enriched_products(
                spark,
                products="non_existent.products",
                save=False
            )
        assert "Source table not found" in str(e.value)

    def test_products_missing_columns_error(self, spark):
        """Test error handling when required columns are missing"""
        incomplete_data = [
            Row(Product_ID="P1", Category="Tech")  # Missing required columns
        ]
        
        incomplete_df = spark.createDataFrame(incomplete_data)
        incomplete_df.write.format("delta").mode("overwrite").saveAsTable("test.incomplete_products")

        with pytest.raises(RuntimeError) as e:
            create_enriched_products(
                spark,
                products="test.incomplete_products",
                save=False
            )
        assert "Products table missing" in str(e.value)

    def test_products_all_valid_records(self, spark):
        """Test scenario where all product records are valid"""
        data = [
            Row(Product_ID="P1", Category="Furniture", Sub_Category="Chairs", 
                Product_Name="Office Chair", State="CA", Price_per_product=200.0),
            Row(Product_ID="P2", Category="Tech", Sub_Category="Phones", 
                Product_Name="Smartphone", State="NY", Price_per_product=500.0),
        ]
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.all_valid_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="test.all_valid_products",
            save=False
        )

        assert valid_df.count() == 2
        assert bad_df.count() == 0

        # Check column renaming
        columns = valid_df.columns
        assert "Product_ID" in columns
        assert "Sub_Category" in columns
        assert "Product_Name" in columns
        assert "Price_Per_Product" in columns
        assert "Created_Timestamp" in columns
        assert "Updated_Timestamp" in columns

    def test_products_all_invalid_records(self, spark):
        """Test scenario where all product records are invalid"""
        data = [
            Row(Product_ID="P1", Category="Tech", Sub_Category="Phones", 
                Product_Name="Phone", State="NY", Price_per_product=-100.0),  # Negative price
            Row(Product_ID="P2", Category="Furniture", Sub_Category="Tables", 
                Product_Name="Table", State="TX", Price_per_product=0.0),     # Zero price
            Row(Product_ID=None, Category="Tech", Sub_Category="Laptops", 
                Product_Name="Laptop", State="CA", Price_per_product=1000.0), # Null Product_ID
        ]
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.all_invalid_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="test.all_invalid_products",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 3

    def test_products_boundary_prices(self, spark):
        """Test products with boundary price values"""
        data = [
            Row(Product_ID="P1", Category="Tech", Sub_Category="Phones", 
                Product_Name="Phone", State="NY", Price_per_product=0.01),    # Very small positive
            Row(Product_ID="P2", Category="Tech", Sub_Category="Phones", 
                Product_Name="Phone", State="NY", Price_per_product=999999.99), # Very large
            Row(Product_ID="P3", Category="Tech", Sub_Category="Phones", 
                Product_Name="Phone", State="NY", Price_per_product=0.0),      # Exactly zero
            Row(Product_ID="P4", Category="Tech", Sub_Category="Phones", 
                Product_Name="Phone", State="NY", Price_per_product=-0.01),    # Very small negative
        ]
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.boundary_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="test.boundary_products",
            save=False
        )

        assert valid_df.count() == 2  # P1 and P2 (positive prices)
        assert bad_df.count() == 2   # P3 and P4 (zero and negative)

        valid_prices = [row["Price_Per_Product"] for row in valid_df.collect()]
        assert 0.01 in valid_prices
        assert 999999.99 in valid_prices

    def test_products_empty_dataframe(self, spark):
        """Test behavior with empty products DataFrame"""
        empty_df = spark.createDataFrame([], 
            StructType([
                StructField("Product ID", StringType(), True),
                StructField("Category", StringType(), True),
                StructField("Sub-Category", StringType(), True),
                StructField("Product Name", StringType(), True),
                StructField("State", StringType(), True),
                StructField("Price per product", DoubleType(), True)
            ]))

        empty_df.write.format("delta").mode("overwrite").saveAsTable("test.empty_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="test.empty_products",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 0

    def test_products_null_handling(self, spark):
        """Test products with various null values"""
        data = [
            Row(Product_ID="P1", Category="Tech", Sub_Category="Phones", 
                Product_Name=None, State="NY", Price_per_product=100.0),      # Null name (should be valid)
            Row(Product_ID="P2", Category=None, Sub_Category="Phones", 
                Product_Name="Phone", State="NY", Price_per_product=200.0),   # Null category (should be valid)
            Row(Product_ID="P3", Category="Tech", Sub_Category=None, 
                Product_Name="Phone", State=None, Price_per_product=300.0),   # Multiple nulls (should be valid)
            Row(Product_ID=None, Category="Tech", Sub_Category="Phones", 
                Product_Name="Phone", State="NY", Price_per_product=400.0),   # Null Product_ID (invalid)
        ]
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.null_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="test.null_products",
            save=False
        )

        assert valid_df.count() == 3  # P1, P2, P3 (only Product_ID null is invalid)
        assert bad_df.count() == 1   # Only the record with null Product_ID

    def test_products_special_characters(self, spark):
        """Test products with special characters in data"""
        data = [
            Row(Product_ID="P@1", Category="Furniture & Décor", Sub_Category="Chairs/Tables", 
                Product_Name="Executive Chair (Leather)", State="CA", Price_per_product=299.99),
        ]
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.special_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="test.special_products",
            save=False
        )

        assert valid_df.count() == 1
        assert bad_df.count() == 0

        row = valid_df.collect()[0]
        assert row["Product_ID"] == "P@1"
        assert row["Category"] == "Furniture & Décor"
        assert row["Sub_Category"] == "Chairs/Tables"
        assert row["Product_Name"] == "Executive Chair (Leather)"

    def test_products_large_dataset(self, spark):
        """Test products processing with larger dataset"""
        data = []
        for i in range(1000):
            price = float(i + 1)  # Prices 1.0 to 1000.0 (all positive)
            data.append(Row(
                Product_ID=f"P{i}",
                Category=f"Category{i % 5}",
                Sub_Category=f"SubCat{i % 3}",
                Product_Name=f"Product {i}",
                State=f"State{i % 10}",
                Price_per_product=price
            ))
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.large_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="test.large_products",
            save=False
        )

        assert valid_df.count() == 1000  # All should be valid
        assert bad_df.count() == 0

    def test_mixed_valid_invalid_products(self, spark):
        """Test mixed scenario with both valid and invalid products"""
        data = [
            # Valid products
            Row(Product_ID="P1", Category="Tech", Sub_Category="Phones", 
                Product_Name="iPhone", State="CA", Price_per_product=999.0),
            Row(Product_ID="P2", Category="Furniture", Sub_Category="Chairs", 
                Product_Name="Office Chair", State="NY", Price_per_product=299.0),
            
            # Invalid products
            Row(Product_ID="P3", Category="Tech", Sub_Category="Laptops", 
                Product_Name="Laptop", State="TX", Price_per_product=-500.0),   # Negative price
            Row(Product_ID=None, Category="Office", Sub_Category="Supplies", 
                Product_Name="Pen", State="FL", Price_per_product=5.0),        # Null Product_ID
            Row(Product_ID="P5", Category="Sports", Sub_Category="Equipment", 
                Product_Name="Ball", State="WA", Price_per_product=0.0),       # Zero price
        ]
        
        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.mixed_products")

        valid_df, bad_df = create_enriched_products(
            spark,
            products="test.mixed_products",
            save=False
        )

        assert valid_df.count() == 2  # P1 and P2
        assert bad_df.count() == 3   # P3, null ID record, and P5

        # Verify valid products
        valid_ids = [row["Product_ID"] for row in valid_df.collect()]
        assert "P1" in valid_ids
        assert "P2" in valid_ids

        # Verify invalid products
        bad_records = bad_df.collect()
        bad_prices = [row["Price_Per_Product"] for row in bad_records if row["Price_Per_Product"] is not None]
        assert -500.0 in bad_prices
        assert 0.0 in bad_prices
