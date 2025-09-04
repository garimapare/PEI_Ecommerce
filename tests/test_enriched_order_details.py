import pytest
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.utils import AnalysisException
import sys
import os
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Dynamically add the repo root to sys.path
repo_root = "/Workspace/Users/pare.garima.18@gmail.com/PEI_Ecommerce/src"
sys.path.append(repo_root)

from src.enriched_order_details import create_enriched_order_details   

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[2]") \
        .appName("pytest-enriched") \
        .getOrCreate()


@pytest.fixture
def setup_tables(spark):
    # Sample Orders
    orders = [
        Row(Order_ID="O1", Order_Date="21/8/2016", Customer_ID="C1", Product_ID="P1", Profit=100.0),
        Row(Order_ID="O2", Order_Date="22/8/2016", Customer_ID=None, Product_ID="P1", Profit=200.0),   # bad (null Customer_ID)
        Row(Order_ID="O3", Order_Date="23/8/2016", Customer_ID="C2", Product_ID="P2", Profit=-50.0)   # bad (negative profit)
    ]
    orders_df = spark.createDataFrame(orders)
    orders_df.write.format("delta").mode("overwrite").saveAsTable("raw.orders")
    orders_df.show()

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
        save=False  # don’t persist in unit tests
    )

    valid = [r.asDict() for r in valid_df.collect()]
    bad = [r.asDict() for r in bad_df.collect()]

    # Check valid record
    assert any(v["Customer_ID"] == "C1" and v["Profit"] == 100.0 for v in valid)

    # Check bad records
    assert any(b["Order_ID"] == "O2" for b in bad)   # missing Customer_ID
    assert any(b["Order_ID"] == "O3" for b in bad)   # negative profit


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


def test_missing_columns(spark, setup_tables):
    # Drop a required column intentionally
    spark.sql("ALTER TABLE raw.orders DROP COLUMN Profit")

    with pytest.raises(RuntimeError) as e:
        create_enriched_order_details(
            spark,
            orders="raw.orders",
            customers="raw.customer",
            products="raw.products",
            save=False
        )
    assert "Orders table missing columns" in str(e.value)



