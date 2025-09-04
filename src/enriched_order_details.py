from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, current_timestamp, lit
from pyspark.sql.utils import AnalysisException


def get_spark(app_name: str = "enriched-order-details") -> SparkSession:
    """
    Get or create a SparkSession.
    - On Databricks: use the active session (Spark Connect).
    - Locally: create a new Spark with master("local[*]").
    """
    spark = SparkSession.getActiveSession()
    if spark:  # Running inside Databricks
        return spark

    # Running locally
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # only used outside Databricks
        .getOrCreate()
    )


def create_enriched_order_details(
    spark: SparkSession,
    orders: str = "raw.orders",
    customers: str = "raw.customer",
    products: str = "raw.products",
    output: str = "silver.order_details",
    error: str = "error.order_details_bad_records",
    save: bool = True,
    job_id: str = "manual_job",
    run_id: str = "manual_run"
):
    """
    Creates an enriched order details table by joining orders, customers, and products.
    Adds audit columns.
    """

    try:
        # ---------------------------
        # Validate if raw tables exist
        # ---------------------------
        for t in [orders, customers, products]:
            if not spark.catalog.tableExists(t):
                raise ValueError(f"Source table not found: {t}")

        # ---------------------------
        # Load raw tables
        # ---------------------------
        orders_df = spark.table(orders)
        customers_df = spark.table(customers)
        products_df = spark.table(products)

        # ---------------------------
        # Validate schema columns
        # ---------------------------
        required_orders = {"Order ID", "Customer ID", "Product ID", "Profit"}
        required_customers = {"Customer ID", "Customer Name", "Country"}
        required_products = {"Product ID", "Category", "Sub-Category"}

        if not required_orders.issubset(set(orders_df.columns)):
            raise ValueError(
                f"Orders table missing columns: {required_orders - set(orders_df.columns)}"
            )
        if not required_customers.issubset(set(customers_df.columns)):
            raise ValueError(
                f"Customers table missing columns: {required_customers - set(customers_df.columns)}"
            )
        if not required_products.issubset(set(products_df.columns)):
            raise ValueError(
                f"Products table missing columns: {required_products - set(products_df.columns)}"
            )

        # ---------------------------------------------------------------
        # Create enriched order details with profit rounded to 2 decimals
        # -----------------------------------------------------------------
        order_details_df = (
            orders_df.alias("o")
            .join(
                customers_df.alias("cust"),
                col("o.Customer ID") == col("cust.Customer ID"),
                "left"
            )
            .join(
                products_df.alias("prod"),
                col("o.Product ID") == col("prod.Product ID"),
                "left"
            )
            .select(
                col("o.Order ID").alias("Order_ID"),
                col("o.Order Date").alias("Order_Date"),
                col("cust.Customer ID").alias("Customer_ID"),
                col("cust.Customer Name").alias("Customer_Name"),
                col("cust.Country"),
                col("prod.Category"),
                col("prod.Sub-Category").alias("Sub_Category"),
                round(col("o.Profit"), 2).alias("Profit"),
            )
        )

        # ---------------------------
        # Add audit columns
        # ---------------------------
        order_details_df = (
            order_details_df
            .withColumn("created_at", current_timestamp())
            .withColumn("modified_at", current_timestamp())
            .withColumn("created_job_id", lit(job_id))
            .withColumn("created_run_id", lit(run_id))
            .withColumn("modified_job_id", lit(job_id))
            .withColumn("modified_run_id", lit(run_id))
        )

        # ---------------------------
        # Data quality checks (DQ Checks)
        # ---------------------------
        valid_order_details_df = order_details_df.filter(
            (col("Customer_ID").isNotNull()) & (col("Profit") >= 0)
        )

        order_details_bad_records_df = order_details_df.exceptAll(valid_order_details_df)

        # ---------------------------
        # Save outputs
        # ---------------------------
        if save:
            valid_order_details_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(
                output
            )
            order_details_bad_records_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(
                error
            )

        return valid_order_details_df, order_details_bad_records_df

    except AnalysisException as ae:
        raise RuntimeError(f"Spark AnalysisException: {str(ae)}")
    except Exception as e:
        raise RuntimeError(f"Failed to create enriched order details: {str(e)}")


def main():
    """Entry point for running as a script"""
    spark = get_spark("enriched-order-details")

    valid_df, bad_df = create_enriched_order_details(
        spark,
        orders="raw.orders",
        customers="raw.customer",
        products="raw.products",
        output="silver.order_details",
        error="error.order_details_bad_records",
        save=True,
        job_id="job_123",
        run_id="run_456"
    )

    print("Enriched order details created successfully")
    print("Valid records:")
    valid_df.show()
    print("Bad records:")
    bad_df.show()

    spark.stop()


if __name__ == "__main__":
    main()