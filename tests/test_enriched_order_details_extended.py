import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import col
import sys
import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Add project root to sys.path
repo_root = "/Workspace/Users/pare.garima.18@gmail.com/PEI_Ecommerce/src"
sys.path.append(repo_root)

from src.enriched_order_details import create_enriched_order_details, get_spark


@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[2]") \
        .appName("pytest-enriched-extended") \
        .getOrCreate()


class TestEnrichedOrderDetailsExtended:
    """Extended test cases for enriched order details functionality"""

    def test_empty_dataframes(self, spark):
        """Test behavior with empty input DataFrames"""
        # Create empty tables with correct schema
        empty_orders = spark.createDataFrame([], 
            StructType([
                StructField("Order_ID", StringType(), True),
                StructField("Order_Date", StringType(), True),
                StructField("Customer_ID", StringType(), True),
                StructField("Product_ID", StringType(), True),
                StructField("Profit", DoubleType(), True)
            ]))
        
        empty_customers = spark.createDataFrame([],
            StructType([
                StructField("Customer_ID", StringType(), True),
                StructField("Customer_Name", StringType(), True),
                StructField("Country", StringType(), True)
            ]))
        
        empty_products = spark.createDataFrame([],
            StructType([
                StructField("Product_ID", StringType(), True),
                StructField("Category", StringType(), True),
                StructField("Sub_Category", StringType(), True)
            ]))

        # Save as temporary tables
        empty_orders.write.format("delta").mode("overwrite").saveAsTable("test.empty_orders")
        empty_customers.write.format("delta").mode("overwrite").saveAsTable("test.empty_customers")
        empty_products.write.format("delta").mode("overwrite").saveAsTable("test.empty_products")

        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders="test.empty_orders",
            customers="test.empty_customers", 
            products="test.empty_products",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 0

    def test_audit_columns_presence(self, spark):
        """Test that audit columns are properly added"""
        # Setup minimal test data
        orders = [Row(Order_ID="O1", Order_Date="21/8/2016", Customer_ID="C1", Product_ID="P1", Profit=100.0)]
        customers = [Row(Customer_ID="C1", Customer_Name="Alice", Country="India")]
        products = [Row(Product_ID="P1", Category="Furniture", Sub_Category="Chairs")]

        spark.createDataFrame(orders).write.format("delta").mode("overwrite").saveAsTable("test.audit_orders")
        spark.createDataFrame(customers).write.format("delta").mode("overwrite").saveAsTable("test.audit_customers")
        spark.createDataFrame(products).write.format("delta").mode("overwrite").saveAsTable("test.audit_products")

        valid_df, _ = create_enriched_order_details(
            spark,
            orders="test.audit_orders",
            customers="test.audit_customers",
            products="test.audit_products",
            save=False,
            job_id="test_job_123",
            run_id="test_run_456"
        )

        # Check audit columns exist
        expected_audit_cols = ["created_at", "modified_at", "created_job_id", "created_run_id", "modified_job_id", "modified_run_id"]
        for col_name in expected_audit_cols:
            assert col_name in valid_df.columns

        # Check audit column values
        row = valid_df.collect()[0]
        assert row["created_job_id"] == "test_job_123"
        assert row["created_run_id"] == "test_run_456"
        assert row["modified_job_id"] == "test_job_123"
        assert row["modified_run_id"] == "test_run_456"

    def test_profit_rounding(self, spark):
        """Test that profit values are properly rounded to 2 decimal places"""
        orders = [
            Row(Order_ID="O1", Order_Date="21/8/2016", Customer_ID="C1", Product_ID="P1", Profit=100.123456),
            Row(Order_ID="O2", Order_Date="22/8/2016", Customer_ID="C1", Product_ID="P1", Profit=200.999)
        ]
        customers = [Row(Customer_ID="C1", Customer_Name="Alice", Country="India")]
        products = [Row(Product_ID="P1", Category="Furniture", Sub_Category="Chairs")]

        spark.createDataFrame(orders).write.format("delta").mode("overwrite").saveAsTable("test.round_orders")
        spark.createDataFrame(customers).write.format("delta").mode("overwrite").saveAsTable("test.round_customers")
        spark.createDataFrame(products).write.format("delta").mode("overwrite").saveAsTable("test.round_products")

        valid_df, _ = create_enriched_order_details(
            spark,
            orders="test.round_orders",
            customers="test.round_customers",
            products="test.round_products",
            save=False
        )

        profits = [row["Profit"] for row in valid_df.collect()]
        assert 100.12 in profits  # 100.123456 rounded to 2 decimals
        assert 201.0 in profits   # 200.999 rounded to 2 decimals

    def test_left_join_behavior(self, spark):
        """Test left join behavior when customers or products are missing"""
        orders = [
            Row(Order_ID="O1", Order_Date="21/8/2016", Customer_ID="C1", Product_ID="P1", Profit=100.0),
            Row(Order_ID="O2", Order_Date="22/8/2016", Customer_ID="C999", Product_ID="P1", Profit=200.0),  # Missing customer
            Row(Order_ID="O3", Order_Date="23/8/2016", Customer_ID="C1", Product_ID="P999", Profit=300.0)   # Missing product
        ]
        customers = [Row(Customer_ID="C1", Customer_Name="Alice", Country="India")]
        products = [Row(Product_ID="P1", Category="Furniture", Sub_Category="Chairs")]

        spark.createDataFrame(orders).write.format("delta").mode("overwrite").saveAsTable("test.join_orders")
        spark.createDataFrame(customers).write.format("delta").mode("overwrite").saveAsTable("test.join_customers")
        spark.createDataFrame(products).write.format("delta").mode("overwrite").saveAsTable("test.join_products")

        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders="test.join_orders",
            customers="test.join_customers",
            products="test.join_products",
            save=False
        )

        # Only O1 should be valid (has both customer and product, positive profit)
        assert valid_df.count() == 1
        valid_row = valid_df.collect()[0]
        assert valid_row["Order_ID"] == "O1"

        # O2 and O3 should be in bad records due to missing joins resulting in null Customer_ID
        assert bad_df.count() == 2

    def test_zero_profit_edge_case(self, spark):
        """Test handling of zero profit (should be valid as it's >= 0)"""
        orders = [Row(Order_ID="O1", Order_Date="21/8/2016", Customer_ID="C1", Product_ID="P1", Profit=0.0)]
        customers = [Row(Customer_ID="C1", Customer_Name="Alice", Country="India")]
        products = [Row(Product_ID="P1", Category="Furniture", Sub_Category="Chairs")]

        spark.createDataFrame(orders).write.format("delta").mode("overwrite").saveAsTable("test.zero_orders")
        spark.createDataFrame(customers).write.format("delta").mode("overwrite").saveAsTable("test.zero_customers")
        spark.createDataFrame(products).write.format("delta").mode("overwrite").saveAsTable("test.zero_products")

        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders="test.zero_orders",
            customers="test.zero_customers",
            products="test.zero_products",
            save=False
        )

        assert valid_df.count() == 1
        assert bad_df.count() == 0
        assert valid_df.collect()[0]["Profit"] == 0.0

    def test_column_name_case_sensitivity(self, spark):
        """Test that column names are handled correctly (case sensitivity)"""
        # Test with different case variations
        orders = [Row(order_id="O1", order_date="21/8/2016", customer_id="C1", product_id="P1", profit=100.0)]
        
        orders_df = spark.createDataFrame(orders)
        orders_df.write.format("delta").mode("overwrite").saveAsTable("test.case_orders")

        customers = [Row(Customer_ID="C1", Customer_Name="Alice", Country="India")]
        products = [Row(Product_ID="P1", Category="Furniture", Sub_Category="Chairs")]

        spark.createDataFrame(customers).write.format("delta").mode("overwrite").saveAsTable("test.case_customers")
        spark.createDataFrame(products).write.format("delta").mode("overwrite").saveAsTable("test.case_products")

        # This should raise an error due to missing required columns
        with pytest.raises(RuntimeError) as e:
            create_enriched_order_details(
                spark,
                orders="test.case_orders",
                customers="test.case_customers",
                products="test.case_products",
                save=False
            )
        assert "Orders table missing columns" in str(e.value)

    def test_get_spark_function(self):
        """Test the get_spark utility function"""
        # Test that get_spark returns a valid SparkSession
        spark_session = get_spark("test-app")
        assert spark_session is not None
        assert spark_session.sparkContext.appName == "test-app"

    def test_large_dataset_simulation(self, spark):
        """Test with a larger dataset to ensure scalability"""
        # Create a larger dataset
        orders = []
        for i in range(1000):
            orders.append(Row(
                Order_ID=f"O{i}",
                Order_Date="21/8/2016",
                Customer_ID=f"C{i % 10}",  # 10 unique customers
                Product_ID=f"P{i % 5}",   # 5 unique products
                Profit=float(i % 100)     # Profits 0-99
            ))

        customers = [Row(Customer_ID=f"C{i}", Customer_Name=f"Customer{i}", Country="TestCountry") for i in range(10)]
        products = [Row(Product_ID=f"P{i}", Category="TestCategory", Sub_Category=f"SubCat{i}") for i in range(5)]

        spark.createDataFrame(orders).write.format("delta").mode("overwrite").saveAsTable("test.large_orders")
        spark.createDataFrame(customers).write.format("delta").mode("overwrite").saveAsTable("test.large_customers")
        spark.createDataFrame(products).write.format("delta").mode("overwrite").saveAsTable("test.large_products")

        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders="test.large_orders",
            customers="test.large_customers",
            products="test.large_products",
            save=False
        )

        # All records should be valid (positive profits, valid joins)
        assert valid_df.count() == 1000
        assert bad_df.count() == 0

    def test_special_characters_in_data(self, spark):
        """Test handling of special characters in string fields"""
        orders = [Row(Order_ID="O'1", Order_Date="21/8/2016", Customer_ID="C&1", Product_ID="P@1", Profit=100.0)]
        customers = [Row(Customer_ID="C&1", Customer_Name="Alice & Bob", Country="Côte d'Ivoire")]
        products = [Row(Product_ID="P@1", Category="Furniture & Décor", Sub_Category="Chairs/Tables")]

        spark.createDataFrame(orders).write.format("delta").mode("overwrite").saveAsTable("test.special_orders")
        spark.createDataFrame(customers).write.format("delta").mode("overwrite").saveAsTable("test.special_customers")
        spark.createDataFrame(products).write.format("delta").mode("overwrite").saveAsTable("test.special_products")

        valid_df, bad_df = create_enriched_order_details(
            spark,
            orders="test.special_orders",
            customers="test.special_customers",
            products="test.special_products",
            save=False
        )

        assert valid_df.count() == 1
        assert bad_df.count() == 0
        
        row = valid_df.collect()[0]
        assert row["Customer_Name"] == "Alice & Bob"
        assert row["Country"] == "Côte d'Ivoire"
        assert row["Category"] == "Furniture & Décor"
