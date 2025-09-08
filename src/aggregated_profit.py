from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, current_timestamp, lit, sum , to_date
)
from pyspark.sql.utils import AnalysisException


def get_spark(app_name: str = "gold-profit-aggregates") -> SparkSession:
    spark = SparkSession.getActiveSession()
    if spark:  # Running inside Databricks
        return spark
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )


def create_gold_profit_aggregates(
    spark: SparkSession,
    silver_orders: str = "silver.order_details",
    output: str = "gold.profit_aggregates",
    error: str = "error.profit_aggregates_bad_records",
    job_id: str = "manual",
    run_id: str = "manual_run",
    save: bool = True,
):
    """
    Creates a gold aggregate table that shows total profit by:
    - Year
    - Product Category
    - Product Sub Category
    - Customer
    Handles incorrect dates and negative profits by sending them to bad records.
    """

    try:
        if not spark.catalog.tableExists(silver_orders):
            raise ValueError(f"Source table not found: {silver_orders}")

        df = spark.table(silver_orders)

        required_cols = {"Order_Date", "Category", "Sub_Category", "Customer_ID", "Profit"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Silver table missing columns: {required_cols - set(df.columns)}")

        # -----------------------------------
        # Add Order_Date column cast to Date
        # -----------------------------------
        df = df.withColumn("Order_Date", to_date(col("Order_Date"), "d/M/yyyy"))

        invalid_dates_df = df.filter(col("Order_Date").isNull())
        valid_dates_df = df.filter(col("Order_Date").isNotNull())

        # ---------------------------
        # Calculate Aggregated Profit
        # ---------------------------
        agg_df = (
            valid_dates_df.withColumn("Year", year(col("Order_Date")))
                          .groupBy("Year", "Category", "Sub_Category", "Customer_ID")
                          .agg(sum("Profit").alias("Total_Profit"))
        )

        # ---------------------------
        # Add audit columns
        # ---------------------------
        enriched_df = (
            agg_df
            .withColumn("created_at", current_timestamp())
            .withColumn("modified_at", current_timestamp())
            .withColumn("created_job_id", lit(job_id))
            .withColumn("created_run_id", lit(run_id))
            .withColumn("modified_job_id", lit(job_id))
            .withColumn("modified_run_id", lit(run_id))
        )

        # ---------------------------
        # Data quality checks
        # ---------------------------
        valid_df = enriched_df.filter(col("Total_Profit") >= 0)
        invalid_profit_df = enriched_df.filter(col("Total_Profit") < 0)

        # Standardize invalid dates schema to match enriched_df
        invalid_dates_df = (
            invalid_dates_df
            .withColumn("Year", lit(None).cast("int"))
            .withColumn("Total_Profit", lit(None).cast("double"))
            .withColumn("created_at", current_timestamp())
            .withColumn("modified_at", current_timestamp())
            .withColumn("created_job_id", lit(job_id))
            .withColumn("created_run_id", lit(run_id))
            .withColumn("modified_job_id", lit(job_id))
            .withColumn("modified_run_id", lit(run_id))
        ).select(valid_df.columns)  # enforce same column order

        bad_df = invalid_profit_df.unionByName(invalid_dates_df)

        # -----------------------------
        # Save outputs to delta tables
        # ------------------------------
        if save:
            # For Delta tables, use createOrReplaceTempView and then CREATE OR REPLACE TABLE
            valid_df.createOrReplaceTempView("temp_valid_profit")
            spark.sql(f"CREATE OR REPLACE TABLE {output} USING DELTA AS SELECT * FROM temp_valid_profit")
            
            bad_df.createOrReplaceTempView("temp_bad_profit")
            spark.sql(f"CREATE OR REPLACE TABLE {error} USING DELTA AS SELECT * FROM temp_bad_profit")

        return valid_df, bad_df

    except AnalysisException as ae:
        raise RuntimeError(f"Spark AnalysisException: {str(ae)}")
    except Exception as e:
        raise RuntimeError(f"Failed to create gold profit aggregates: {str(e)}")


def main():
    spark = get_spark()
    if spark is None:
        raise RuntimeError("No active Spark session found. Run inside Databricks workspace.")

    valid_df, bad_df = create_gold_profit_aggregates(
        spark,
        silver_orders="silver.order_details",
        output="gold.profit_aggregates",
        error="error.profit_aggregates_bad_records",
        job_id="gold_profit_agg_job",
        run_id="12345",
        save=True,
    )

    print("Gold profit aggregates created successfully")
    print("Valid Records:")
    valid_df.show()
    print("Bad Records:")
    bad_df.show()

    spark.stop()


if __name__ == "__main__":
    main()
