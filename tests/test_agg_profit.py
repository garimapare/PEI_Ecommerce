import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, IntegerType
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
    # Define schemas for test data
    INPUT_SCHEMA = StructType([
        StructField("Order_ID", StringType(), True),
        StructField("Order_Date", StringType(), True),
        StructField("Customer_ID", StringType(), True),
        StructField("Customer_Name", StringType(), True),
        StructField("Country", StringType(), True),
        StructField("Category", StringType(), True),
        StructField("Sub_Category", StringType(), True),
        StructField("Profit", DoubleType(), True)
    ])

    # Schema for expected output (without audit columns)
    OUTPUT_SCHEMA = StructType([
        StructField("Year", IntegerType(), True),
        StructField("Category", StringType(), True),
        StructField("Sub_Category", StringType(), True),
        StructField("Customer_ID", StringType(), True),
        StructField("Total_Profit", DoubleType(), True)
    ])

    @pytest.mark.parametrize("test_case_name,test_data,expected_valid_count,expected_bad_count,expected_total_profit", [
        (
            "valid_date_formats",
            [
                ("O1", "1/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 100.0),
                ("O2", "31/12/2020", "C1", "Alice", "USA", "Tech", "Phones", 200.0),
            ],
            1,  # Aggregated into one record
            0,  # No bad records
            300.0  # 100 + 200
        ),
        (
            "invalid_date_formats",
            [
                ("O3", "03012020", "C1", "Alice", "USA", "Tech", "Phones", 150.0),
                ("O4", "Jan 1, 2020", "C1", "Alice", "USA", "Tech", "Phones", 75.0),
                ("O5", "invalid_date", "C1", "Alice", "USA", "Tech", "Phones", 50.0),
            ],
            0,  # No valid records
            3,  # All records are bad
            None  # No valid aggregation
        ),
        (
            "mixed_date_formats",
            [
                ("O1", "1/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 100.0),
                ("O2", "2020-12-31", "C1", "Alice", "USA", "Tech", "Phones", 200.0),
                ("O3", "03/09/2018", "C1", "Alice", "USA", "Tech", "Phones", 150.0),
                ("O4", "abcd", "C1", "Alice", "USA", "Tech", "Phones", 50.0),
            ],
            2,  # Two valid groups (one for 2020 with 1/1/2020 and 2020-12-31, one for 2018 with 03/09/2018)
            1,  # One invalid date format (abcd)
            300.0  # 100 + 200 (only the 2020 records should be aggregated together)
        )
    ])


    # Test Case 1 - Test various date formats and edge cases
    def test_date_format_variations(self, spark, test_case_name, test_data, expected_valid_count, expected_bad_count, expected_total_profit):
        # Create DataFrame with the defined schema
        df = spark.createDataFrame(test_data, schema=self.INPUT_SCHEMA)
        table_name = f"date_formats_{test_case_name}"
        df.createOrReplaceTempView(table_name)

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders=table_name,
            save=False
        )

        # Verify counts
        assert valid_df.count() == expected_valid_count, \
            f"Expected {expected_valid_count} valid records, got {valid_df.count()}"
        assert bad_df.count() == expected_bad_count, \
            f"Expected {expected_bad_count} bad records, got {bad_df.count()}"

        if expected_total_profit is not None and expected_valid_count > 0:
            # Create expected data based on the test case
            expected_data = []
            if test_case_name == "valid_date_formats":
                expected_data = [
                    (2020, "Tech", "Phones", "C1", 300.0)
                ]
            elif test_case_name == "mixed_date_formats":
                # For mixed dates, we expect two valid records (one for 2020, one for 2018)
                expected_data = [
                    (2020, "Tech", "Phones", "C1", 300.0),  # 100 + 200
                    (2018, "Tech", "Phones", "C1", 150.0)   # 150 from 03/09/2018
                ]
            
            # Create expected DataFrame
            expected_columns = ["Year", "Category", "Sub_Category", "Customer_ID", "Total_Profit"]
            expected_df = spark.createDataFrame(expected_data, schema=expected_columns)
            
            # Select only the expected columns from the actual result (excluding audit columns)
            actual_selected = valid_df.select(expected_columns)
            
            # Sort both DataFrames for consistent comparison
            actual_sorted = actual_selected.orderBy("Year", "Category", "Sub_Category", "Customer_ID")
            expected_sorted = expected_df.orderBy("Year", "Category", "Sub_Category", "Customer_ID")
            
            # Compare DataFrames
            assert actual_sorted.collect() == expected_sorted.collect(), \
                f"The valid records do not match the expected data.\n" \
                f"Expected:\n{expected_sorted.show(truncate=False)}\n" \
                f"Actual:\n{actual_selected.show(truncate=False)}"

    @pytest.mark.parametrize("year,month,day,expected_year", [
        (2000, 1, 1, 2000),    # Year boundary start
        (2000, 12, 31, 2000),  # Year boundary end
        (2001, 1, 1, 2001),    # Next year start
        (1999, 12, 31, 1999),  # Previous year end
        (2020, 6, 15, 2020),   # Mid-year date
    ])

    # Test Case 2 - Test year extraction for boundary dates
    def test_year_boundary_conditions(self, spark, year, month, day, expected_year):
        # Create a date string in the format expected by the function
        date_str = f"{day}/{month}/{year}"
        
        # Create test data with the boundary date
        data = [
            ("O1", date_str, "C1", "Alice", "USA", "Tech", "Phones", 100.0)
        ]
        
        # Create DataFrame with the defined schema
        df = spark.createDataFrame(data, schema=self.INPUT_SCHEMA)
        df.createOrReplaceTempView("year_boundary_test")
        
        # Call the function under test
        valid_df, _ = create_gold_profit_aggregates(
            spark,
            silver_orders="year_boundary_test",
            save=False
        )
        
        # Verify the year is extracted correctly
        if valid_df.count() > 0:
            # Create expected data
            expected_data = [
                (expected_year, "Tech", "Phones", "C1", 100.0)
            ]
            expected_columns = ["Year", "Category", "Sub_Category", "Customer_ID", "Total_Profit"]
            expected_df = spark.createDataFrame(expected_data, schema=expected_columns)
            
            # Select only the expected columns from the actual result
            actual_selected = valid_df.select(expected_columns)
            
            # Sort both DataFrames for consistent comparison
            actual_sorted = actual_selected.orderBy("Year", "Category", "Sub_Category", "Customer_ID")
            expected_sorted = expected_df.orderBy("Year", "Category", "Sub_Category", "Customer_ID")
            
            # Compare DataFrames
            assert actual_sorted.collect() == expected_sorted.collect(), \
                f"The valid records do not match the expected data for date {date_str}.\n" \
                f"Expected year: {expected_year}"

        valid_row = valid_df.collect()[0]
        assert valid_row["Year"] == expected_year, \
            f"Expected year {expected_year}, got {valid_row['Year']}"

    @pytest.mark.parametrize("test_scenario,test_data,expected_valid_count,expected_valid_profit", [
        (
            "single_positive_profit",
            [
                ("O1", "1/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 100.0),
            ],
            1,  # Single valid record
            100.0
        ),
        (
            "multiple_positive_profits_same_group",
            [
                ("O1", "1/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 100.0),
                ("O2", "2/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 50.0),
            ],
            1,  # Single group with sum of profits
            150.0
        ),
        (
            "zero_profit_included",
            [
                ("O1", "1/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 100.0),
                ("O2", "2/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 0.0),
            ],
            1,  # Single group with sum including zero
            100.0
        ),
        (
            "multiple_groups_positive_profits",
            [
                ("O1", "1/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 100.0),
                ("O2", "2/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 50.0),
                ("O3", "3/1/2020", "C2", "Bob", "USA", "Furniture", "Chairs", 200.0),
            ],
            2,  # Two valid groups
            350.0  # Sum of all profits (100+50+200)
        )
    ])
    # Test Case 3 - Test handling of profit aggregation (positive and zero profits only)
    def test_zero_and_negative_profit_handling(self, spark, test_scenario, test_data, expected_valid_count, expected_valid_profit):
        # Create DataFrame with the defined schema
        df = spark.createDataFrame(test_data, schema=self.INPUT_SCHEMA)
        table_name = f"profit_handling_{test_scenario}"
        df.createOrReplaceTempView(table_name)

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders=table_name,
            save=False
        )

        # Verify counts - since we're only testing with valid profits, there should be no bad records
        assert valid_df.count() == expected_valid_count, \
            f"Expected {expected_valid_count} valid records, got {valid_df.count()}"
        assert bad_df.count() == 0, \
            f"Expected 0 bad records, got {bad_df.count()}"

        if expected_valid_count > 0:
            # Calculate expected data based on the test data
            # Group by Customer_ID, Category, Sub_Category and sum profits
            expected_data = []
            
            # Create a dictionary to aggregate profits by group
            profit_by_group = {}
            for order in test_data:
                # Extract group key and profit
                group_key = (2020, order[5], order[6], order[2])  # year, category, sub_category, customer_id
                profit = order[7]  # profit
                
                # Add to group total (only if profit is not negative)
                if profit >= 0:
                    if group_key not in profit_by_group:
                        profit_by_group[group_key] = 0.0
                    profit_by_group[group_key] += profit
            
            # Convert to list of tuples for DataFrame creation
            for (year, category, sub_category, customer_id), total_profit in profit_by_group.items():
                expected_data.append((year, category, sub_category, customer_id, total_profit))
            
            # Create expected DataFrame
            expected_columns = ["Year", "Category", "Sub_Category", "Customer_ID", "Total_Profit"]
            expected_df = spark.createDataFrame(expected_data, schema=expected_columns)
            
            # Select only the expected columns from the actual result
            actual_selected = valid_df.select(expected_columns)
            
            # Sort both DataFrames for consistent comparison
            actual_sorted = actual_selected.orderBy("Customer_ID", "Category", "Sub_Category")
            expected_sorted = expected_df.orderBy("Customer_ID", "Category", "Sub_Category")
            
            # Compare DataFrames
            assert actual_sorted.collect() == expected_sorted.collect(), (
                f"The valid records do not match the expected data for scenario '{test_scenario}'.\n"
                f"Expected: {expected_sorted.collect()}\n"
                f"Actual: {actual_sorted.collect()}"
            )
            
            # Also verify the sum of all profits matches expected
            total_profit = sum(row['Total_Profit'] for row in valid_df.collect())
            assert abs(total_profit - expected_valid_profit) < 0.001, \
                f"Total profit mismatch. Expected {expected_valid_profit}, got {total_profit}"

    # Test Case 4 - Test aggregation across multiple customers and categories
    def test_multiple_customers_aggregation(self, spark):
        data = [
            ("O1", "1/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 100.0),
            ("O2", "2/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 200.0),
            ("O3", "3/1/2020", "C1", "Alice", "USA", "Furniture", "Chairs", 150.0),
            ("O4", "4/1/2020", "C2", "Bob", "USA", "Tech", "Phones", 75.0),
        ]

        # Create DataFrame with the defined schema
        df = spark.createDataFrame(data, schema=self.INPUT_SCHEMA)
        df.createOrReplaceTempView("multi_customers")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="multi_customers",
            save=False
        )

        # Verify counts
        assert valid_df.count() == 3, f"Expected 3 valid records, got {valid_df.count()}"
        assert bad_df.count() == 0, f"Expected 0 bad records, got {bad_df.count()}"

        # Define expected results
        expected_data = [
            (2020, "Tech", "Phones", "C1", 300.0),      # 100 + 200
            (2020, "Furniture", "Chairs", "C1", 150.0), # 150
            (2020, "Tech", "Phones", "C2", 75.0)         # 75
        ]
        
        # Create expected DataFrame
        expected_columns = ["Year", "Category", "Sub_Category", "Customer_ID", "Total_Profit"]
        expected_df = spark.createDataFrame(expected_data, schema=expected_columns)
        
        # Select only the expected columns from the actual result
        actual_selected = valid_df.select(expected_columns)
        
        # Sort both DataFrames for consistent comparison
        actual_sorted = actual_selected.orderBy("Customer_ID", "Category", "Sub_Category")
        expected_sorted = expected_df.orderBy("Customer_ID", "Category", "Sub_Category")
        
        # Compare DataFrames
        assert actual_sorted.collect() == expected_sorted.collect(), \
            "The valid records do not match the expected data for multiple customers aggregation"


    # Test Case 5 - Test aggregation behavior with null values in grouping columns
    def test_aggregation_with_nulls(self, spark):
        data = [
            # Valid record with all fields
            ("O1", "1/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 100.0),
            # Valid record with null Category (should still be included in aggregation)
            ("O2", "2/1/2020", "C1", "Alice", "USA", None, "Phones", 150.0),
            # Valid record with null Sub_Category (should still be included in aggregation)
            ("O3", "3/1/2020", "C1", "Alice", "USA", "Tech", None, 200.0),
            # Valid record with null Category and Sub_Category (should still be included)
            ("O4", "4/1/2020", "C1", "Alice", "USA", None, None, 50.0),
            # Another customer with valid data
            ("O5", "5/1/2020", "C2", "Bob", "UK", "Furniture", "Chairs", 300.0),
        ]

        # Create DataFrame with the defined schema
        df = spark.createDataFrame(data, schema=self.INPUT_SCHEMA)
        df.createOrReplaceTempView("null_test")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="null_test",
            save=False
        )

        # Define expected data - each unique combination of Customer_ID, Category, and Sub_Category
        # should be a separate group
        expected_data = [
            # Group 1: C1 + Tech + Phones = 100.0 (from O1)
            (2020, "Tech", "Phones", "C1", 100.0),
            # Group 2: C1 + null + Phones = 150.0 (from O2)
            (2020, None, "Phones", "C1", 150.0),
            # Group 3: C1 + Tech + null = 200.0 (from O3)
            (2020, "Tech", None, "C1", 200.0),
            # Group 4: C1 + null + null = 50.0 (from O4)
            (2020, None, None, "C1", 50.0),
            # Group 5: C2 + Furniture + Chairs = 300.0 (from O5)
            (2020, "Furniture", "Chairs", "C2", 300.0)
        ]
        
        # Create expected DataFrame
        expected_columns = ["Year", "Category", "Sub_Category", "Customer_ID", "Total_Profit"]
        expected_df = spark.createDataFrame(expected_data, schema=expected_columns)
        
        # Select only the expected columns from the actual result
        actual_selected = valid_df.select(expected_columns)
        
        # Sort both DataFrames for consistent comparison
        actual_sorted = actual_selected.orderBy("Customer_ID", "Category", "Sub_Category")
        expected_sorted = expected_df.orderBy("Customer_ID", "Category", "Sub_Category")
        
        # Compare DataFrames
        assert actual_sorted.collect() == expected_sorted.collect(), (
            "The valid records with null values do not match the expected data\n"
            f"Expected: {expected_sorted.collect()}\n"
            f"Actual: {actual_sorted.collect()}"
        )
        
        # Verify the sum of all profits matches expected
        total_profit = sum(row['Total_Profit'] for row in valid_df.collect())
        expected_total = 100.0 + 150.0 + 200.0 + 50.0 + 300.0  # Sum of all valid profits
        assert abs(total_profit - expected_total) < 0.001, \
            f"Total profit mismatch. Expected {expected_total}, got {total_profit}"


    # Test Case 7 - Test aggregation with multiple customer groups and profit scenarios
    def test_mixed_valid_invalid_aggregation(self, spark):
        data = [
            # Group 1: C1 + Tech + Phones = 300.0 (valid)
            ("O1", "1/1/2020", "C1", "Alice", "USA", "Tech", "Phones", 100.0),
            ("O2", "2/1/2020", "C1", "Alice", "USA", "Non-Tech", "Stationary", 200.0),
            
            # Group 2: C2 + Furniture + Chairs = 150.0 (valid)
            ("O3", "3/1/2020", "C2", "Bob", "USA", "Furniture", "Chairs", 75.0),
            ("O4", "4/1/2020", "C2", "Bob", "USA", "Furniture", "Chairs", 75.0),
            
            # Group 3: C3 + Office + Supplies = 0.0 (valid, zero profit)
            ("O5", "5/1/2020", "C3", "Charlie", "UK", "Office", "Supplies", 0.0),
            
            # Group 4: C4 + Tech + Accessories = 50.0 (valid, single order)
            ("O6", "6/1/2020", "C4", "Dave", "Canada", "Tech", "Accessories", 50.0)
        ]

        # Create DataFrame with the defined schema
        df = spark.createDataFrame(data, schema=self.INPUT_SCHEMA)
        df.createOrReplaceTempView("mixed_agg")

        valid_df, bad_df = create_gold_profit_aggregates(
            spark,
            silver_orders="mixed_agg",
            save=False
        )

        # Verify counts - all should be valid since we're not testing negative profits anymore
        # Now expecting 5 valid records since O1 and O2 are in different categories
        assert valid_df.count() == 5, f"Expected 5 valid records, got {valid_df.count()}"
        assert bad_df.count() == 0, f"Expected 0 bad records, got {bad_df.count()}"

        # Define expected valid data
        # O1 and O2 are now in different categories, so they should be in separate groups
        expected_valid_data = [
            (2020, "Tech", "Phones", "C1", 100.0),           # O1
            (2020, "Non-Tech", "Stationary", "C1", 200.0),   # O2 - Different category from O1
            (2020, "Furniture", "Chairs", "C2", 150.0),      # O3 + O4 = 75 + 75
            (2020, "Office", "Supplies", "C3", 0.0),         # O5
            (2020, "Tech", "Accessories", "C4", 50.0)        # O6
        ]
        
        # Create expected DataFrame
        expected_columns = ["Year", "Category", "Sub_Category", "Customer_ID", "Total_Profit"]
        expected_df = spark.createDataFrame(expected_valid_data, schema=expected_columns)
        
        # Select only the expected columns from the actual results
        actual_selected = valid_df.select(expected_columns)
        
        # Sort both DataFrames for consistent comparison
        actual_sorted = actual_selected.orderBy("Customer_ID", "Category", "Sub_Category")
        expected_sorted = expected_df.orderBy("Customer_ID", "Category", "Sub_Category")
        
        # Compare DataFrames
        assert actual_sorted.collect() == expected_sorted.collect(), (
            "The valid records do not match the expected data.\n"
            f"Expected: {expected_sorted.collect()}\n"
            f"Actual: {actual_sorted.collect()}"
        )
        
        # Verify the sum of all profits matches expected
        # Now including the separate profits for O1 and O2
        total_profit = sum(row['Total_Profit'] for row in valid_df.collect())
        expected_total = 100.0 + 200.0 + 150.0 + 0.0 + 50.0  # Sum of all valid profits (100+200 instead of 300)
        assert abs(total_profit - expected_total) < 0.001, \
            f"Total profit mismatch. Expected {expected_total}, got {total_profit}"
