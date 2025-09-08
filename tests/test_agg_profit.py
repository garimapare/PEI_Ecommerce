import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
import sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Add project root to sys.path
# import sys, os
print("PYTHONPATH:", sys.path)
print("Files in src:", os.listdir(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.aggregated_profit import create_gold_profit_aggregates

@pytest.fixture(scope="session")
def spark():
    """Provide a Spark session for tests (Databricks or local)."""
    # If Spark already exists (Databricks), return it
    spark = SparkSession.getActiveSession()
    if spark:
        return spark

    # Else run locally with Delta Lake configuration
    return (
        SparkSession.builder
        .appName("pytest-gold-profit-aggregates")
        .master("local[2]")
        .config("spark.jars.packages", "io.delta:delta-spark_2.13:3.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


@pytest.fixture
def setup_silver_table(spark):
    """Prepare a mock silver.order_details table with edge cases."""
    data = [
        # validate profits
        Row(Order_ID="O1", Order_Date="2016-08-21", Customer_ID="C1",
            Customer_Name="Alice", Country="India", Category="Furniture",
            Sub_Category="Chairs", Profit=100.0),

        Row(Order_ID="O2", Order_Date="2016-08-22", Customer_ID="C1",
            Customer_Name="Alice", Country="India", Category="Furniture",
            Sub_Category="Chairs", Profit=200.0),

        # validate different category/sub-category
        Row(Order_ID="O3", Order_Date="2016-09-01", Customer_ID="C2",
            Customer_Name="Bob", Country="USA", Category="Technology",
            Sub_Category="Phones", Profit=300.0),

        # Validate negative profit (bad record)
        Row(Order_ID="O4", Order_Date="2017-01-05", Customer_ID="C2",
            Customer_Name="Bob", Country="USA", Category="Technology",
            Sub_Category="Phones", Profit=-50.0),

        # validate incorrect date (edge case → should drop into bad records)
        Row(Order_ID="O5", Order_Date="21/8/2016", Customer_ID="C3",
            Customer_Name="Eve", Country="UK", Category="Office Supplies",
            Sub_Category="Paper", Profit=400.0),
    ]

    df = spark.createDataFrame(data)
    # Use Delta Lake compatible table creation
    df.createOrReplaceTempView("temp_silver_order_details")
    spark.sql("CREATE OR REPLACE TABLE silver.order_details USING DELTA AS SELECT * FROM temp_silver_order_details")


def collect_as_set(df, cols):
    """Helper to collect DF rows as set of dicts for comparison."""
    return {tuple([r[c] for c in cols]) for r in df.collect()}


def test_valid_aggregation(spark, setup_silver_table):
    valid_df, bad_df = create_gold_profit_aggregates(
        spark,
        silver_orders="silver.order_details",
        save=False
    )

    # expected valid aggregations
    expected_valid = {
        (2016, "Furniture", "Chairs", "C1", 300.0),   # aggregated 100 + 200
        (2016, "Technology", "Phones", "C2", 300.0), # valid record only
    }

    actual_valid = collect_as_set(valid_df, ["Year", "Category", "Sub_Category", "Customer_ID", "Total_Profit"])
    assert expected_valid == actual_valid


def test_negative_and_incorrect_records(spark, setup_silver_table):
    valid_df, bad_df = create_gold_profit_aggregates(
        spark,
        silver_orders="silver.order_details",
        save=False
    )

    # negative profit (O4) and malformed date (O5) should go to bad records
    actual_bad = collect_as_set(bad_df, ["Customer_ID", "Category", "Sub_Category", "Total_Profit"])
    
    assert any(row[3] == -50.0 for row in actual_bad)  # negative profit
    assert any(row[3] == 400.0 for row in actual_bad)  # malformed date case


def test_missing_table_raises(spark):
    with pytest.raises(RuntimeError) as e:
        create_gold_profit_aggregates(
            spark,
            silver_orders="silver.non_existing",
            save=False
        )
    assert "Source table not found" in str(e.value)


def test_missing_columns(spark, setup_silver_table):
    # Drop column intentionally
    spark.sql("ALTER TABLE silver.order_details DROP COLUMN Profit")

    with pytest.raises(RuntimeError) as e:
        create_gold_profit_aggregates(
            spark,
            silver_orders="silver.order_details",
            save=False
        )
    assert "Silver table missing columns" in str(e.value)
