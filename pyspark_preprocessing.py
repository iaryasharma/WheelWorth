# pyspark_preprocessing.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when, lit, year
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

def create_spark_session(app_name="Car Price Predictor"):
    """Create or get a Spark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

def clean_numeric_column(df, column_name):
    """Clean numeric columns by removing non-numeric characters."""
    return df.withColumn(
        column_name,
        when(col(column_name).isNull(), None)
        .otherwise(
            col(column_name).cast(StringType())
        )
    ).withColumn(
        column_name,
        regexp_replace(col(column_name), "[^0-9.]", "").cast(DoubleType())
    )

def preprocess_data(spark, data_path):
    """Load and preprocess the car data for model training."""
    # Load data
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    # Print schema and sample data
    print("\nOriginal schema:")
    df.printSchema()
    print("\nSample data:")
    df.show(5, truncate=False)
    
    # Clean price column - remove currency symbols, commas, etc.
    if 'AskPrice' in df.columns:
        df = df.withColumn('AskPrice', regexp_replace(col('AskPrice'), '[₹,]', ''))
        df = clean_numeric_column(df, 'AskPrice')
    
    # Clean kmDriven - remove non-numeric characters
    if 'kmDriven' in df.columns:
        df = clean_numeric_column(df, 'kmDriven')
    
    # Convert Year to numeric
    if 'Year' in df.columns:
        df = df.withColumn('Year', col('Year').cast(IntegerType()))
    
    # Handle Age column
    if 'Age' in df.columns:
        df = df.withColumn('Age', col('Age').cast(IntegerType()))
    else:
        current_yr = 2025  # Current year
        df = df.withColumn('Age', lit(current_yr) - col('Year'))
    
    # Drop rows with missing values in critical columns
    critical_columns = ['Brand', 'model', 'Year', 'kmDriven', 'AskPrice']
    for column in critical_columns:
        if column in df.columns:
            df = df.filter(col(column).isNotNull())
    
    # Fill missing values for non-critical columns
    df = df.fillna({
        'Transmission': 'Manual',
        'Owner': 'first',
        'FuelType': 'Petrol',
        'PostedDate': 'Unknown',
        'AdditionInfo': 'No additional info'
    })
    
    print("\nPreprocessed schema:")
    df.printSchema()
    print("\nPreprocessed sample data:")
    df.show(5, truncate=False)
    
    return df

def create_feature_pipeline(df):
    """Create a feature preprocessing pipeline for ML."""
    # Identify categorical and numeric columns
    categorical_cols = [field.name for field in df.schema.fields 
                       if field.dataType == StringType() and field.name != 'AskPrice']
    numeric_cols = [field.name for field in df.schema.fields 
                   if (field.dataType == IntegerType() or field.dataType == DoubleType()) 
                   and field.name != 'AskPrice']
    
    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")
    
    # Create indexers for categorical columns
    indexers = [StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep") 
                for col_name in categorical_cols]
    
    # Create encoders for indexed columns
    encoders = [OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_vec")
                for col_name in categorical_cols]
    
    # Assemble all features into a vector
    assembler_inputs = [f"{col_name}_vec" for col_name in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    
    # Create a standard scaler
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    
    # Create the pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    
    return pipeline, categorical_cols, numeric_cols

def prepare_data(data_path):
    """Prepare data for model training and evaluation."""
    # Create Spark session
    spark = create_spark_session()
    
    # Preprocess data
    df = preprocess_data(spark, data_path)
    
    # Create feature pipeline
    pipeline, categorical_cols, numeric_cols = create_feature_pipeline(df)
    
    # Fit the pipeline to the data
    print("\nFitting feature pipeline...")
    model = pipeline.fit(df)
    
    # Transform the data
    print("\nTransforming data...")
    transformed_df = model.transform(df)
    
    # Select relevant columns for model training
    final_df = transformed_df.select("scaled_features", "AskPrice")
    
    # Split data into training and test sets
    train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)
    
    print(f"\nTraining set count: {train_df.count()}")
    print(f"Test set count: {test_df.count()}")
    
    return train_df, test_df, model, spark, df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        train_df, test_df, feature_pipeline, spark, original_df = prepare_data(data_path)
        print("Data preparation successful!")
    else:
        print("Please provide a data path as argument")