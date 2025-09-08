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
    """Provide a Spark session for tests"""
    # In Databricks, use the existing session or create without specifying master
    existing_session = SparkSession.getActiveSession()
    if existing_session:
        return existing_session
    else:
        # Fallback for local development
        return SparkSession.builder \
            .appName("pytest-gold-profit-extended") \
            .getOrCreate()

class TestAggregatedProfitExtended:
    """Extended test cases for aggregated profit functionality"""

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
                Row(Order_ID="O3", Order_Date="2020-01-01", Customer_ID="C1", Customer_Name="Alice", 
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
                Row(Order_ID="O3", Order_Date="2020-01-01", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=150.0),
                Row(Order_ID="O4", Order_Date="invalid_date", Customer_ID="C1", Customer_Name="Alice", 
                    Country="USA", Category="Tech", Sub_Category="Phones", Profit=50.0),
            ],
            1,  # Only valid dates aggregated
            2,  # Invalid date formats
            300.0  # 100 + 200
        )
    ])
    def test_date_format_variations(self, spark, test_case_name, test_data, expected_valid_count, expected_bad_count, expected_total_profit):
        """Test various date formats and edge cases"""
        df = spark.createDataFrame(test_data)
        table_name = f"test.date_formats_{test_case_name}"
        df.write.format("delta").mode("overwrite").saveAsTable(table_name)

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
    def test_year_boundary_conditions(self, spark, year, month, day, expected_year):
        """Test year extraction for boundary dates"""
        data = [
            Row(Order_ID="O1", Order_Date=f"{day}/{month}/{year}", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
        ]

        df = spark.createDataFrame(data)
        table_name = f"test.year_boundary_{year}_{month}_{day}"
        df.write.format("delta").mode("overwrite").saveAsTable(table_name)

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
    def test_zero_and_negative_profit_handling(self, spark, test_scenario, test_data, expected_valid_count, expected_bad_count, expected_valid_profit):
        """Test handling of zero and negative profits"""
        df = spark.createDataFrame(test_data)
        table_name = f"test.profit_handling_{test_scenario}"
        df.write.format("delta").mode("overwrite").saveAsTable(table_name)

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
    def test_schema_validation_comprehensive(self, spark, missing_columns, expected_error_message):
        """Test comprehensive schema validation"""
        # Create a base row with all columns
        base_data = {
            "Order_ID": "O1",
            "Order_Date": "1/1/2020",
            "Customer_ID": "C1",
            "Customer_Name": "Alice",
            "Country": "USA",
            "Category": "Tech",
            "Sub_Category": "Phones",
            "Profit": 100.0
        }
        
        # Remove the specified columns
        for col in missing_columns:
            base_data.pop(col, None)
        
        incomplete_df = spark.createDataFrame([Row(**base_data)])
        table_name = f"test.incomplete_schema_{'_'.join(missing_columns)}"
        incomplete_df.write.format("delta").mode("overwrite").saveAsTable(table_name)

        with pytest.raises(RuntimeError) as e:
            create_gold_profit_aggregates(
                spark,
                silver_orders=table_name,
                save=False
            )
        assert expected_error_message in str(e.value)

    def test_multiple_customers_aggregation(self, spark):
        """Test aggregation across multiple customers and categories"""
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
        df.write.format("delta").mode("overwrite").saveAsTable("test.multi_customers")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="test.multi_customers",
            save=False
        )

        # Should have 3 aggregated records
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

    def test_empty_silver_table(self, spark):
        """Test behavior with empty silver table"""
        empty_df = spark.createDataFrame([], 
            StructType([
                StructField("Order_ID", StringType(), True),
                StructField("Order_Date", StringType(), True),
                StructField("Customer_ID", StringType(), True),
                StructField("Customer_Name", StringType(), True),
                StructField("Country", StringType(), True),
                StructField("Category", StringType(), True),
                StructField("Sub_Category", StringType(), True),
                StructField("Profit", DoubleType(), True)
            ]))

        empty_df.write.format("delta").mode("overwrite").saveAsTable("test.empty_silver")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="test.empty_silver",
            save=False
        )

        assert valid_df.count() == 0
        assert bad_df.count() == 0

    def test_aggregation_with_nulls(self, spark):
        """Test aggregation behavior with null values in grouping columns"""
        data = [
            Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0),
            Row(Order_ID="O2", Order_Date="2/1/2020", Customer_ID=None, Customer_Name="Unknown", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=200.0),
            Row(Order_ID="O3", Order_Date="3/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category=None, Sub_Category="Phones", Profit=150.0),
        ]

        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.nulls_agg")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="test.nulls_agg",
            save=False
        )

        # All should be processed (nulls are valid in groupBy)
        # Should have 3 different groups due to null values
        assert valid_df.count() == 3
        assert bad_df.count() == 0

    def test_get_spark_utility(self):
        """Test the get_spark utility function"""
        spark_session = get_spark("test-gold-app")
        assert spark_session is not None
    
        # In Databricks, we get the existing session, so app name might be different
        # In local mode, we create a new session with the specified app name
        active_session = SparkSession.getActiveSession()
        if active_session and active_session == spark_session:
            # We're in Databricks - just verify we got a valid session
            assert spark_session.sparkContext is not None
        else:
            # We're in local mode - verify the app name was set correctly
            assert spark_session.sparkContext.appName == "test-gold-app"


    def test_large_aggregation_dataset(self, spark):
        """Test aggregation with larger dataset"""
        # Create 1000 records across 10 customers, 5 categories, 2 years
        data = []
        for i in range(1000):
            year = 2020 + (i % 2)  # 2020 or 2021
            month = (i % 12) + 1
            day = (i % 28) + 1
            
            data.append(Row(
                Order_ID=f"O{i}",
                Order_Date=f"{day}/{month}/{year}",
                Customer_ID=f"C{i % 10}",
                Customer_Name=f"Customer{i % 10}",
                Country="TestCountry",
                Category=f"Category{i % 5}",
                Sub_Category=f"SubCat{i % 3}",
                Profit=float(i % 50 + 1)  # Profits 1-50
            ))

        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.large_agg")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="test.large_agg",
            save=False
        )

        # Should have many aggregated records (10 customers × 5 categories × 3 sub_categories × 2 years = max 300)
        # But actual count depends on the modulo distribution
        assert valid_df.count() > 0
        assert bad_df.count() == 0

        # Verify aggregation logic with a sample
        sample_agg = valid_df.filter(
            (col("Customer_ID") == "C0") & 
            (col("Category") == "Category0") & 
            (col("Sub_Category") == "SubCat0") &
            (col("Year") == 2020)
        ).collect()
        
        if sample_agg:  # If this combination exists
            assert len(sample_agg) == 1  # Should be aggregated into one record

    def test_audit_columns_in_aggregation(self, spark):
        """Test that audit columns are properly added in aggregation"""
        data = [
            Row(Order_ID="O1", Order_Date="1/1/2020", Customer_ID="C1", Customer_Name="Alice", 
                Country="USA", Category="Tech", Sub_Category="Phones", Profit=100.0)
        ]

        df = spark.createDataFrame(data)
        df.write.format("delta").mode("overwrite").saveAsTable("test.audit_agg")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="test.audit_agg",
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

    def test_mixed_valid_invalid_aggregation(self, spark):
        """Test aggregation where some groups are valid and others invalid"""
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
        df.write.format("delta").mode("overwrite").saveAsTable("test.mixed_agg")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="test.mixed_agg",
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
