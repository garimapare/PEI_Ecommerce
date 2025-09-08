"""
Utility functions for Delta Lake compatible test operations
"""

def create_or_replace_delta_table(spark, df, table_name):
    """
    Create or replace a Delta table in a way that's compatible with Delta Lake.
    
    Args:
        spark: SparkSession instance
        df: DataFrame to save
        table_name: Name of the table to create/replace
    """
    # Use a unique temp view name to avoid conflicts
    temp_view_name = f"temp_{table_name.replace('.', '_').replace('-', '_')}"
    
    # Create temp view and then use CREATE OR REPLACE TABLE
    df.createOrReplaceTempView(temp_view_name)
    spark.sql(f"CREATE OR REPLACE TABLE {table_name} USING DELTA AS SELECT * FROM {temp_view_name}")
    
    # Clean up temp view
    spark.catalog.dropTempView(temp_view_name)
