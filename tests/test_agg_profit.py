import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
from pyspark.sql.functions import col, to_date
import sys
import os
from datetime import date

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.aggregated_profit import create_gold_profit_aggregates, get_spark


@pytest.fixture(scope="session")
def spark():
    # In Databricks, use the existing session or create without specifying master
    existing_session = SparkSession.getActiveSession()
    if existing_session:
        return existing_session
    else:
        # Fallback for local development with basic Spark configuration
        return SparkSession.builder \
            .appName("pytest-gold-profit-extended") \
            .master("local[*]") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.ansi.enabled", "false") \
            .config("spark.sql.storeAssignmentPolicy", "LEGACY") \
            .getOrCreate()

class TestAggregatedProfitExtended:

    @pytest.mark.parametrize("test_case_name,test_data,expected_valid_count,expected_bad_count,expected_total_profit", [
        (
            "valid_date_formats",
            [
                Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
                Row(Order_ID="O2", Order_Date="31/12/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=200.0),
            ],
            1,  # Aggregated into one record
            0,  # No bad records
            300.0  # 100 + 200
        ),
        (
            "invalid_date_formats",
            [
                Row(Order_ID="O3", Order_Date="invalid_iso_format", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=150.0),
                Row(Order_ID="O4", Order_Date="Jan 1, 2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=75.0),
                Row(Order_ID="O5", Order_Date="invalid_date", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=50.0),
            ],
            0,  # No valid records
            3,  # All records are bad
            None  # No valid aggregation
        ),
        (
            "mixed_date_formats",
            [
                Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
                Row(Order_ID="O2", Order_Date="31/12/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=200.0),
                Row(Order_ID="O3", Order_Date="invalid_iso_date", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=150.0),
                Row(Order_ID="O4", Order_Date="invalid_date", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=50.0),
            ],
            1,  # Only valid dates aggregated
            2,  # Invalid date formats
            300.0  # 100 + 200
        )
    ])


    # Test Case 1 - Test various date formats and edge cases
    def test_date_format_variations(self, spark, test_case_name, test_data, expected_valid_count, expected_bad_count, expected_total_profit):
        df = spark.createDataFrame(test_data)
        table_name = f"date_formats_{test_case_name}"
        df.createOrReplaceTempView(table_name)

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders=table_name,
            save=False
        )

        assert valid_df.count() == expected_valid_count
        assert bad_df.count() == expected_bad_count

        if expected_total_profit is not None and expected_valid_count > 0:
            valid_row = valid_df.collect()[0]
            assert valid_row["Year"] == 2020
            assert valid_row["Total_Profit"] == expected_total_profit

    @pytest.mark.parametrize("year,month,day,expected_year", [
        (2000, 1, 1, 2000),    # Year boundary start
        (2000, 12, 31, 2000),  # Year boundary end
        (2001, 1, 1, 2001),    # Next year start
        (1999, 12, 31, 1999),  # Previous year end
        (2020, 6, 15, 2020),   # Mid-year date
    ])

    # Test Case 2 - Test year extraction for boundary dates
    def test_year_boundary_conditions(self, spark, year, month, day, expected_year):
        data = [
            Row(Order_ID="O1", Order_Date=f"{day}/{month}/{year}", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
        ]

        df = spark.createDataFrame(data)
        table_name = f"year_boundary_{year}_{month}_{day}"
        df.createOrReplaceTempView(table_name)

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders=table_name,
            save=False
        )

        assert valid_df.count() == 1
        assert bad_df.count() == 0

        valid_row = valid_df.collect()[0]
        assert valid_row["Year"] == expected_year

    @pytest.mark.parametrize("test_scenario,test_data,expected_valid_count,expected_bad_count,expected_valid_profit", [
        (
            "zero_profit_included",
            [
                Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
                Row(Order_ID="O2", Order_Date="2/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=0.0),
            ],
            1,  # C1 group is valid (100.0 + 0.0 = 100.0)
            0,  # No bad records
            100.0
        ),
        (
            "negative_total_excluded",
            [
                Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=-50.0),
                Row(Order_ID="O2", Order_Date="2/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=-25.0),
            ],
            0,  # No valid records (total is negative)
            1,  # C1 group is bad
            None
        ),
        (
            "mixed_positive_negative",
            [
                Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
                Row(Order_ID="O2", Order_Date="2/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=-30.0),
                Row(Order_ID="O3", Order_Date="3/1/2020", Customer_ID="C2", Customer_Name="Bob", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=-100.0),
            ],
            1,  # C1 group is valid (100.0 - 30.0 = 70.0)
            1,  # C2 group is bad (-100.0)
            70.0
        )
    ])
    # Test Case 3 - Test handling of zero and negative profits
    def test_zero_and_negative_profit_handling(self, spark, test_scenario, test_data, expected_valid_count, expected_bad_count, expected_valid_profit):
        df = spark.createDataFrame(test_data)
        table_name = f"profit_handling_{test_scenario}"
        df.createOrReplaceTempView(table_name)

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders=table_name,
            save=False
        )

        assert valid_df.count() == expected_valid_count
        assert bad_df.count() == expected_bad_count

        if expected_valid_profit is not None and expected_valid_count > 0:
            valid_row = valid_df.collect()[0]
            assert valid_row["Total_Profit"] == expected_valid_profit

    @pytest.mark.parametrize("missing_columns,expected_error_message", [
        (
            ["Order_Date", "Category", "Sub_Category"],
            "Silver table missing columns"
        ),
        (
            ["Customer_ID", "Customer_Name"],
            "Silver table missing columns"
        ),
        (
            ["Profit"],
            "Silver table missing columns"
        )
    ])

    # Test Case 5 - Test aggregation across multiple customers and categories
    def test_multiple_customers_aggregation(self, spark):
        data = [
            # Customer C1 - Tech category
            Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
            Row(Order_ID="O2", Order_Date="2/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=200.0),
            
            # Customer C1 - Furniture category
            Row(Order_ID="O3", Order_Date="3/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Furniture", Sub_Category="Chairs", Profit=150.0),
            
            # Customer C2 - Tech category
            Row(Order_ID="O4", Order_Date="4/1/2020", Customer_ID="C2", Customer_Name="Bob", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=75.0),
        ]

        df = spark.createDataFrame(data)
        df.createOrReplaceTempView("multi_customers")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="multi_customers",
            save=False
        )

        # Should have 3 aggregated records and 0 bad records
        assert valid_df.count() == 3
        assert bad_df.count() == 0

        # Convert to dict for easier verification
        results = {}
        for row in valid_df.collect():
            key = (row["Customer_ID"], row["Category"], row["Sub_Category"])
            results[key] = row["Total_Profit"]

        assert results[("C1", "Tech", "Phones")] == 300.0      # 100 + 200
        assert results[("C1", "Furniture", "Chairs")] == 150.0
        assert results[("C2", "Tech", "Phones")] == 75.0


    # Test Case 6 - Test aggregation behavior with null values in grouping columns
    def test_aggregation_with_nulls(self, spark):
        data = [
            Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
            Row(Order_ID="O2", Order_Date="2/1/2020", Customer_ID=None, Customer_Name="Unknown", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=200.0),
            Row(Order_ID="O3", Order_Date="3/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category=None, Sub_Category="Phones", Profit=150.0),
        ]

        df = spark.createDataFrame(data)
        df.createOrReplaceTempView("null_agg")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="null_agg",
            save=False
        )

        # All should be processed (nulls are valid in groupBy)
        # Should have 3 different groups due to null values
        assert valid_df.count() == 3
        assert bad_df.count() == 0

    # Test Case 7 - Test that audit columns are properly added in aggregation
    def test_audit_columns_in_aggregation(self, spark):
        data = [
            Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0)
        ]

        df = spark.createDataFrame(data)
        df.createOrReplaceTempView("audit_agg")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="audit_agg",
            job_id="test_job_agg",
            run_id="test_run_agg",
            save=False
        )

        # Check audit columns exist
        expected_audit_cols = ["created_at", "modified_at", "created_job_id", "created_run_id", "modified_job_id", "modified_run_id"]
        for col_name in expected_audit_cols:
            assert col_name in valid_df.columns

        # Check audit column values
        row = valid_df.collect()[0]
        assert row["created_job_id"] == "test_job_agg"
        assert row["created_run_id"] == "test_run_agg"

    # Test Case 8 - Test aggregation where some groups are valid and others invalid
    def test_mixed_valid_invalid_aggregation(self, spark):
        data = [
            # Valid group: C1 + Tech + Phones = 300.0 (positive)
            Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
            Row(Order_ID="O2", Order_Date="2/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=200.0),
            
            # Invalid group: C2 + Furniture + Chairs = -50.0 (negative total)
            Row(Order_ID="O3", Order_Date="3/1/2020", Customer_ID="C2", Customer_Name="Bob", 
                Country="USA", Category="Furniture", Sub_Category="Chairs", Profit=-30.0),
            Row(Order_ID="O4", Order_Date="4/1/2020", Customer_ID="C2", Customer_Name="Bob", 
                Country="USA", Category="Furniture", Sub_Category="Chairs", Profit=-20.0),
        ]

        df = spark.createDataFrame(data)
        df.createOrReplaceTempView("mixed_agg")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="mixed_agg",
            save=False
        )

        assert valid_df.count() == 1  # Only C1 group is valid
        assert bad_df.count() == 1   # C2 group is invalid

        valid_row = valid_df.collect()[0]
        assert valid_row["Customer_ID"] == "C1"
        assert valid_row["Total_Profit"] == 300.0

        bad_row = bad_df.collect()[0]
        assert bad_row["Customer_ID"] == "C2"
        assert bad_row["Total_Profit"] == -50.0
