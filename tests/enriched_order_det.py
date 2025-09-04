import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.utils import AnalysisException
from chispa.dataframe_comparer import assert_df_equality  # for comparing DataFrames
import sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Dynamically add the repo root to sys.path
repo_root = "/Workspace/Users/pare.garima.18@gmail.com/PEI_Ecommerce/src"
sys.path.append(repo_root)

from src.enriched_order_details import create_enriched_order_details   


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("pytest-enriched")
        .getOrCreate()
    )


@pytest.fixture
def setup_tables(spark):
    # Sample Orders
    orders = [
        Row(Order_ID="O1", Order_Date="21/8/2016", Customer_ID="C1", Product_ID="P1", Profit=100.0),
        Row(Order_ID="O2", Order_Date="22/8/2016", Customer_ID=None, Product_ID="P1", Profit=200.0),  # bad
        Row(Order_ID="O3", Order_Date="23/8/2016", Customer_ID="C2", Product_ID="P2", Profit=-50.0)  # bad
    ]
    orders_df = spark.createDataFrame(orders)
    orders_df.write.format("delta").mode("overwrite").saveAsTable("raw.orders")

    # Sample Customers
    customers = [
        Row(Customer_ID="C1", Customer_Name="Alice", Country="India"),
        Row(Customer_ID="C2", Customer_Name="Bob", Country="USA")
    ]
    customers_df = spark.createDataFrame(customers)
    customers_df.write.format("delta").mode("overwrite").saveAsTable("raw.customer")

    # Sample Products
    products = [
        Row(Product_ID="P1", Category="Furniture", Sub_Category="Chairs"),
        Row(Product_ID="P2", Category="Office Supplies", Sub_Category="Paper")
    ]
    products_df = spark.createDataFrame(products)
    products_df.write.format("delta").mode("overwrite").saveAsTable("raw.products")


def test_valid_and_invalid_records(spark, setup_tables):
    valid_df, bad_df = create_enriched_order_details(
        spark,
        orders="raw.orders",
        customers="raw.customer",
        products="raw.products",
        save=False
    )

    # Expected valid rows
    expected_valid = spark.createDataFrame([
        Row(Order_ID="O1", Order_Date="21/8/2016", Customer_ID="C1",
            Customer_Name="Alice", Country="India", Category="Furniture",
            Sub_Category="Chairs", Profit=100.0)
    ])

    # Expected bad rows (O2: null customer, O3: negative profit)
    expected_bad = spark.createDataFrame([
        Row(Order_ID="O2", Order_Date="22/8/2016", Customer_ID=None,
            Customer_Name="Alice", Country="India", Category="Furniture",
            Sub_Category="Chairs", Profit=200.0),
        Row(Order_ID="O3", Order_Date="23/8/2016", Customer_ID="C2",
            Customer_Name="Bob", Country="USA", Category="Office Supplies",
            Sub_Category="Paper", Profit=-50.0)
    ])

    assert_df_equality(valid_df, expected_valid, ignore_row_order=True, ignore_column_order=True)
    assert_df_equality(bad_df, expected_bad, ignore_row_order=True, ignore_column_order=True)


def test_missing_table_raises(spark):
    with pytest.raises(RuntimeError) as e:
        create_enriched_order_details(
            spark,
            orders="raw.non_existing",
            customers="raw.customer",
            products="raw.products",
            save=False
        )
    assert "Source table not found" in str(e.value)


def test_expected_schema(spark, setup_tables):
    """Instead of dropping columns, validate schema expectation."""
    valid_df, _ = create_enriched_order_details(
        spark,
        orders="raw.orders",
        customers="raw.customer",
        products="raw.products",
        save=False
    )

    expected_cols = {
        "Order_ID", "Order_Date", "Customer_ID", "Customer_Name",
        "Country", "Category", "Sub_Category", "Profit"
    }
    assert set(valid_df.columns) == expected_cols
