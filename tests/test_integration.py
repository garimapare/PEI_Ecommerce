import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
import sys
import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.enriched_order_details import create_enriched_order_details
from src.aggregated_profit import create_gold_profit_aggregates
from src.enriched_customers_products import create_enriched_customers, create_enriched_products


@pytest.fixture(scope="session")
def spark():
    """Provide a Spark session for integration tests"""
    return SparkSession.builder \
        .appName("pytest-integration") \
        .master("local[2]") \
        .getOrCreate()


@pytest.fixture
def setup_complete_pipeline_data(spark):
    """Setup complete test data for end-to-end pipeline testing"""
    
    # Raw Orders
    orders_data = [
        Row(Order_ID="O1", Order_Date="21/8/2016", Customer_ID="C1", Product_ID="P1", Profit=100.0),
        Row(Order_ID="O2", Order_Date="22/8/2016", Customer_ID="C1", Product_ID="P2", Profit=200.0),
        Row(Order_ID="O3", Order_Date="23/8/2017", Customer_ID="C2", Product_ID="P1", Profit=150.0),
        Row(Order_ID="O4", Order_Date="24/8/2017", Customer_ID="C2", Product_ID="P3", Profit=75.0),
        # Bad records
        Row(Order_ID="O5", Order_Date="25/8/2016", Customer_ID=None, Product_ID="P1", Profit=50.0),  # Null customer
        Row(Order_ID="O6", Order_Date="26/8/2016", Customer_ID="C1", Product_ID="P1", Profit=-25.0), # Negative profit
    ]
    
    # Raw Customers
    customers_data = [
        Row(Customer_ID="C1", Customer_Name="Alice Johnson", Email="alice@test.com", 
            Phone="123-456-7890", Address="123 Main St", Segment="Consumer", 
            Country="USA", City="New York", State="NY", Postal_Code="10001", Region="East"),
        Row(Customer_ID="C2", Customer_Name="Bob Smith", Email="bob@test.com", 
            Phone="987-654-3210", Address="456 Oak Ave", Segment="Corporate", 
            Country="USA", City="Los Angeles", State="CA", Postal_Code="90001", Region="West"),
        Row(Customer_ID="C3", Customer_Name="Charlie Brown", Email="charlie@test.com", 
            Phone="555-123-4567", Address="789 Pine St", Segment="Consumer", 
            Country="Canada", City="Toronto", State="ON", Postal_Code="M5V3A8", Region="North"),
    ]
    
    # Raw Products
    products_data = [
        Row(Product_ID="P1", Category="Furniture", Sub_Category="Chairs", 
            Product_Name="Executive Chair", State="CA", Price_per_product=299.99),
        Row(Product_ID="P2", Category="Technology", Sub_Category="Phones", 
            Product_Name="Smartphone", State="NY", Price_per_product=699.99),
        Row(Product_ID="P3", Category="Office Supplies", Sub_Category="Paper", 
            Product_Name="Copy Paper", State="TX", Price_per_product=29.99),
    ]
    
    # Create and save tables
    spark.createDataFrame(orders_data).write.format("delta").mode("overwrite").saveAsTable("integration.raw_orders")
    spark.createDataFrame(customers_data).write.format("delta").mode("overwrite").saveAsTable("integration.raw_customers")
    spark.createDataFrame(products_data).write.format("delta").mode("overwrite").saveAsTable("integration.raw_products")
    
    return {
        "orders": "integration.raw_orders",
        "customers": "integration.raw_customers", 
        "products": "integration.raw_products"
    }


class TestEndToEndPipeline:
    """Integration tests for the complete data pipeline"""
    
    def test_complete_pipeline_flow(self, spark, setup_complete_pipeline_data):
        """Test the complete pipeline from raw data to gold aggregates"""
        
        # Step 1: Create enriched order details (Bronze -> Silver)
        valid_orders, bad_orders = create_enriched_order_details(
            spark,
            orders=setup_complete_pipeline_data["orders"],
            customers=setup_complete_pipeline_data["customers"],
            products=setup_complete_pipeline_data["products"],
            output="integration.silver_order_details",
            error="integration.error_order_details",
            save=True,
            job_id="integration_test",
            run_id="run_001"
        )
        
        # Verify silver layer results
        assert valid_orders.count() == 4  # O1, O2, O3, O4 should be valid
        assert bad_orders.count() == 2    # O5 (null customer), O6 (negative profit)
        
        # Step 2: Create enriched customers (Bronze -> Silver)
        valid_customers, bad_customers = create_enriched_customers(
            spark,
            customers=setup_complete_pipeline_data["customers"],
            output="integration.silver_customers",
            error="integration.error_customers",
            save=True
        )
        
        assert valid_customers.count() == 3  # All customers should be valid
        assert bad_customers.count() == 0
        
        # Step 3: Create enriched products (Bronze -> Silver)
        valid_products, bad_products = create_enriched_products(
            spark,
            products=setup_complete_pipeline_data["products"],
            output="integration.silver_products", 
            error="integration.error_products",
            save=True
        )
        
        assert valid_products.count() == 3  # All products should be valid
        assert bad_products.count() == 0
        
        # Step 4: Create gold profit aggregates (Silver -> Gold)
        valid_aggregates, bad_aggregates = create_gold_profit_aggregates(
            spark,
            silver_orders="integration.silver_order_details",
            output="integration.gold_profit_aggregates",
            error="integration.error_profit_aggregates", 
            job_id="integration_test",
            run_id="run_001",
            save=True
        )
        
        # Verify gold layer results
        # Should have aggregates by Year, Category, Sub_Category, Customer_ID
        assert valid_aggregates.count() > 0
        assert bad_aggregates.count() == 0  # No bad records expected from valid silver data
        
        # Verify specific aggregations
        agg_results = valid_aggregates.collect()
        
        # Check that we have expected aggregations
        customer_totals = {}
        for row in agg_results:
            key = (row["Customer_ID"], row["Year"], row["Category"], row["Sub_Category"])
            customer_totals[key] = row["Total_Profit"]
        
        # Verify some expected aggregations exist
        assert len(customer_totals) > 0
        
        # Check audit columns are present in all layers
        for df in [valid_orders, valid_customers, valid_products, valid_aggregates]:
            assert "created_at" in df.columns
            assert "modified_at" in df.columns

    def test_pipeline_with_data_quality_issues(self, spark):
        """Test pipeline behavior when data quality issues are present at multiple stages"""
        
        # Create problematic data
        orders_data = [
            Row(Order_ID="O1", Order_Date="invalid_date", Customer_ID="C1", Product_ID="P1", Profit=100.0),
            Row(Order_ID="O2", Order_Date="21/8/2016", Customer_ID="C999", Product_ID="P1", Profit=200.0),  # Missing customer
            Row(Order_ID="O3", Order_Date="22/8/2016", Customer_ID="C1", Product_ID="P999", Profit=150.0),  # Missing product
        ]
        
        customers_data = [
            Row(Customer_ID="C1", Customer_Name="Alice", Email="alice@test.com", 
                Phone="123", Address="Addr1", Segment="Consumer", Country="USA", 
                City="NY", State="NY", Postal_Code="10001", Region="East"),
            Row(Customer_ID=None, Customer_Name="Invalid Customer", Email="invalid@test.com", 
                Phone="456", Address="Addr2", Segment="Consumer", Country="USA", 
                City="SF", State="CA", Postal_Code="94016", Region="West"),  # Invalid customer
        ]
        
        products_data = [
            Row(Product_ID="P1", Category="Tech", Sub_Category="Phones", 
                Product_Name="Phone", State="CA", Price_per_product=500.0),
            Row(Product_ID="P2", Category="Tech", Sub_Category="Phones", 
                Product_Name="Phone", State="NY", Price_per_product=-100.0),  # Invalid product
        ]
        
        # Save test data
        spark.createDataFrame(orders_data).write.format("delta").mode("overwrite").saveAsTable("integration.bad_orders")
        spark.createDataFrame(customers_data).write.format("delta").mode("overwrite").saveAsTable("integration.bad_customers")
        spark.createDataFrame(products_data).write.format("delta").mode("overwrite").saveAsTable("integration.bad_products")
        
        # Test each stage handles bad data appropriately
        
        # Orders enrichment
        valid_orders, bad_orders = create_enriched_order_details(
            spark,
            orders="integration.bad_orders",
            customers="integration.bad_customers",
            products="integration.bad_products",
            save=False
        )
        
        # Only O1 might be valid if it joins successfully, but others should be bad
        assert bad_orders.count() > 0
        
        # Customer enrichment
        valid_customers, bad_customers = create_enriched_customers(
            spark,
            customers="integration.bad_customers",
            save=False
        )
        
        assert valid_customers.count() == 1  # Only C1 is valid
        assert bad_customers.count() == 1   # Null Customer_ID record
        
        # Product enrichment
        valid_products, bad_products = create_enriched_products(
            spark,
            products="integration.bad_products",
            save=False
        )
        
        assert valid_products.count() == 1  # Only P1 is valid
        assert bad_products.count() == 1   # P2 has negative price

    def test_pipeline_performance_with_larger_dataset(self, spark):
        """Test pipeline performance and correctness with larger datasets"""
        
        # Generate larger test dataset
        orders_data = []
        for i in range(5000):
            orders_data.append(Row(
                Order_ID=f"O{i}",
                Order_Date=f"{(i % 28) + 1}/{(i % 12) + 1}/201{(i % 10)}",
                Customer_ID=f"C{i % 100}",  # 100 unique customers
                Product_ID=f"P{i % 50}",   # 50 unique products
                Profit=float((i % 1000) + 1)  # Profits 1-1000
            ))
        
        customers_data = []
        for i in range(100):
            customers_data.append(Row(
                Customer_ID=f"C{i}",
                Customer_Name=f"Customer {i}",
                Email=f"customer{i}@test.com",
                Phone=f"123-456-{i:04d}",
                Address=f"{i} Test St",
                Segment="Consumer" if i % 2 == 0 else "Corporate",
                Country="USA",
                City=f"City{i % 10}",
                State=f"ST{i % 5}",
                Postal_Code=f"{10000 + i}",
                Region=f"Region{i % 4}"
            ))
        
        products_data = []
        for i in range(50):
            products_data.append(Row(
                Product_ID=f"P{i}",
                Category=f"Category{i % 5}",
                Sub_Category=f"SubCat{i % 10}",
                Product_Name=f"Product {i}",
                State=f"ST{i % 5}",
                Price_per_product=float((i % 500) + 1)
            ))
        
        # Save large datasets
        spark.createDataFrame(orders_data).write.format("delta").mode("overwrite").saveAsTable("integration.large_orders")
        spark.createDataFrame(customers_data).write.format("delta").mode("overwrite").saveAsTable("integration.large_customers")
        spark.createDataFrame(products_data).write.format("delta").mode("overwrite").saveAsTable("integration.large_products")
        
        # Test pipeline with large data
        valid_orders, bad_orders = create_enriched_order_details(
            spark,
            orders="integration.large_orders",
            customers="integration.large_customers",
            products="integration.large_products",
            save=False
        )
        
        # All records should be valid with this generated data
        assert valid_orders.count() == 5000
        assert bad_orders.count() == 0
        
        # Test aggregation with large dataset
        # First save the valid orders to use in aggregation
        valid_orders.write.format("delta").mode("overwrite").saveAsTable("integration.large_silver_orders")
        
        valid_aggregates, bad_aggregates = create_gold_profit_aggregates(
            spark,
            silver_orders="integration.large_silver_orders",
            save=False
        )
        
        # Should have many aggregated records
        assert valid_aggregates.count() > 0
        assert bad_aggregates.count() == 0

    def test_pipeline_data_lineage_and_audit(self, spark, setup_complete_pipeline_data):
        """Test that audit columns and data lineage are properly maintained throughout pipeline"""
        
        job_id = "lineage_test_job"
        run_id = "lineage_test_run"
        
        # Create enriched order details with specific job/run IDs
        valid_orders, _ = create_enriched_order_details(
            spark,
            orders=setup_complete_pipeline_data["orders"],
            customers=setup_complete_pipeline_data["customers"],
            products=setup_complete_pipeline_data["products"],
            save=False,
            job_id=job_id,
            run_id=run_id
        )
        
        # Verify audit columns
        sample_row = valid_orders.collect()[0]
        assert sample_row["created_job_id"] == job_id
        assert sample_row["created_run_id"] == run_id
        assert sample_row["modified_job_id"] == job_id
        assert sample_row["modified_run_id"] == run_id
        assert sample_row["created_at"] is not None
        assert sample_row["modified_at"] is not None
        
        # Save for next stage
        valid_orders.write.format("delta").mode("overwrite").saveAsTable("integration.lineage_silver_orders")
        
        # Create gold aggregates with different job/run IDs
        gold_job_id = "gold_lineage_job"
        gold_run_id = "gold_lineage_run"
        
        valid_aggregates, _ = create_gold_profit_aggregates(
            spark,
            silver_orders="integration.lineage_silver_orders",
            job_id=gold_job_id,
            run_id=gold_run_id,
            save=False
        )
        
        # Verify gold layer audit columns
        gold_sample = valid_aggregates.collect()[0]
        assert gold_sample["created_job_id"] == gold_job_id
        assert gold_sample["created_run_id"] == gold_run_id
        assert gold_sample["modified_job_id"] == gold_job_id
        assert gold_sample["modified_run_id"] == gold_run_id

    def test_pipeline_idempotency(self, spark, setup_complete_pipeline_data):
        """Test that running the pipeline multiple times produces consistent results"""
        
        # Run pipeline first time
        valid_orders_1, bad_orders_1 = create_enriched_order_details(
            spark,
            orders=setup_complete_pipeline_data["orders"],
            customers=setup_complete_pipeline_data["customers"],
            products=setup_complete_pipeline_data["products"],
            save=False
        )
        
        # Run pipeline second time with same data
        valid_orders_2, bad_orders_2 = create_enriched_order_details(
            spark,
            orders=setup_complete_pipeline_data["orders"],
            customers=setup_complete_pipeline_data["customers"],
            products=setup_complete_pipeline_data["products"],
            save=False
        )
        
        # Results should be identical (excluding timestamp columns)
        assert valid_orders_1.count() == valid_orders_2.count()
        assert bad_orders_1.count() == bad_orders_2.count()
        
        # Compare business data (excluding audit timestamps)
        business_cols = ["Order_ID", "Customer_ID", "Customer_Name", "Country", "Category", "Sub_Category", "Profit"]
        
        valid_1_business = valid_orders_1.select(*business_cols).collect()
        valid_2_business = valid_orders_2.select(*business_cols).collect()
        
        # Sort both results for comparison
        valid_1_sorted = sorted([row.asDict() for row in valid_1_business], key=lambda x: x["Order_ID"])
        valid_2_sorted = sorted([row.asDict() for row in valid_2_business], key=lambda x: x["Order_ID"])
        
        assert valid_1_sorted == valid_2_sorted
