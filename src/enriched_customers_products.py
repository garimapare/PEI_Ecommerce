from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp
from pyspark.sql.utils import AnalysisException

def create_enriched_customers(
    spark: SparkSession,
    customers: str = "raw.customer",
    output: str = "silver.customers",
    error: str = "error.customers_bad_records",
    save: bool = True
):
    try:
        if not spark.catalog.tableExists(customers):
            raise ValueError(f"Source table not found: {customers}")

        customers_df = spark.table(customers)

        # Required columns
        required_customers = {
            "Customer ID", "Customer Name", "email", "phone", "address",
            "Segment", "Country", "City", "State", "Postal Code", "Region"
        }

        if not required_customers.issubset(set(customers_df.columns)):
            raise ValueError(f"Customers table missing: {required_customers - set(customers_df.columns)}")

        # Rename columns for consistency
        customers_df = customers_df.select(
            col("Customer ID").alias("Customer_ID"),
            col("Customer Name").alias("Customer_Name"),
            col("Email"),
            col("Phone"),
            col("Address"),
            col("Segment"),
            col("Country"),
            col("City"),
            col("State"),
            col("Postal Code").alias("Postal_Code"),
            col("Region"),
            current_timestamp().alias("Created_Timestamp"),
            current_timestamp().alias("Updated_Timestamp")
        )

        # Validation: Customer_ID must not be null
        valid_customers_df = customers_df.filter(col("Customer_ID").isNotNull())
        bad_customers_df = customers_df.exceptAll(valid_customers_df)

        if save:
            # For Delta tables, use createOrReplaceTempView and then CREATE OR REPLACE TABLE
            valid_customers_df.createOrReplaceTempView("temp_valid_customers")
            spark.sql(f"CREATE OR REPLACE TABLE {output} USING DELTA AS SELECT * FROM temp_valid_customers")
            
            bad_customers_df.createOrReplaceTempView("temp_bad_customers")
            spark.sql(f"CREATE OR REPLACE TABLE {error} USING DELTA AS SELECT * FROM temp_bad_customers")

        return valid_customers_df, bad_customers_df

    except AnalysisException as ae:
        raise RuntimeError(f"Spark AnalysisException: {str(ae)}")
    except Exception as e:
        raise RuntimeError(f"Failed to create enriched customers: {str(e)}")


def create_enriched_products(
    spark: SparkSession,
    products: str = "raw.products",
    output: str = "silver.products",
    error: str = "error.products_bad_records",
    save: bool = True
):
    try:
        if not spark.catalog.tableExists(products):
            raise ValueError(f"Source table not found: {products}")

        products_df = spark.table(products)

        # Required columns
        required_products = {"Product ID", "Category", "Sub-Category", "Product Name", "State", "Price per product"}

        if not required_products.issubset(set(products_df.columns)):
            raise ValueError(f"Products table missing: {required_products - set(products_df.columns)}")

        # Rename columns for consistency
        products_df = products_df.select(
            col("Product ID").alias("Product_ID"),
            col("Category"),
            col("Sub-Category").alias("Sub_Category"),
            col("Product Name").alias("Product_Name"),
            col("State"),
            col("Price per product").alias("Price_Per_Product"),
            current_timestamp().alias("Created_Timestamp"),
            current_timestamp().alias("Updated_Timestamp")
        )

        # Validations
        valid_products_df = products_df.filter(
            (col("Product_ID").isNotNull()) &
            (col("Price_Per_Product") > 0)
        )
        bad_products_df = products_df.exceptAll(valid_products_df)

        if save:
            # For Delta tables, use createOrReplaceTempView and then CREATE OR REPLACE TABLE
            valid_products_df.createOrReplaceTempView("temp_valid_products")
            spark.sql(f"CREATE OR REPLACE TABLE {output} USING DELTA AS SELECT * FROM temp_valid_products")
            
            bad_products_df.createOrReplaceTempView("temp_bad_products")
            spark.sql(f"CREATE OR REPLACE TABLE {error} USING DELTA AS SELECT * FROM temp_bad_products")

        return valid_products_df, bad_products_df

    except AnalysisException as ae:
        raise RuntimeError(f"Spark AnalysisException: {str(ae)}")
    except Exception as e:
        raise RuntimeError(f"Failed to create enriched products: {str(e)}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("enriched_customers_products").getOrCreate()

    valid_customers_df, bad_customers_df = create_enriched_customers(
        spark,
        customers="raw.customer",
        output="silver.customers",
        error="error.customers_bad_records",
        save=True
    )

    valid_products_df, bad_products_df = create_enriched_products(
        spark,
        products="raw.products",
        output="silver.products",
        error="error.products_bad_records",
        save=True
    )

    valid_customers_df.show()
    valid_products_df.show()