import os
import json
import time
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel

def create_spark_session(app_name="Car Price Predictor"):
    """Create or get a Spark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

def load_models():
    """Load the trained model and feature pipeline."""
    try:
        # Create spark session
        spark = create_spark_session()
        
        # Load metadata
        metadata_path = os.path.join('models', 'model_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load feature pipeline
        pipeline_path = os.path.join('models', 'feature_pipeline')
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Feature pipeline not found at {pipeline_path}")
        
        feature_pipeline = PipelineModel.load(pipeline_path)
        
        # Load model
        model_path = os.path.join('models', 'car_price_model')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load the appropriate model type based on metadata
        model_name = metadata["model_name"]
        if model_name == "Linear Regression":
            model = LinearRegressionModel.load(model_path)
        elif model_name == "Random Forest":
            model = RandomForestRegressionModel.load(model_path)
        elif model_name == "Gradient Boosting":
            model = GBTRegressionModel.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        print(f"Successfully loaded {model_name} model and feature pipeline")
        
        return spark, model, feature_pipeline, metadata
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def format_input_data(spark, input_data):
    """Format input data as a Spark DataFrame."""
    # Define schema based on expected input
    schema = StructType([
        StructField("Brand", StringType(), True),
        StructField("model", StringType(), True),
        StructField("Year", IntegerType(), True),
        StructField("Age", IntegerType(), True),
        StructField("kmDriven", DoubleType(), True),
        StructField("Transmission", StringType(), True),
        StructField("Owner", StringType(), True),
        StructField("FuelType", StringType(), True),
        StructField("PostedDate", StringType(), True),
        StructField("AdditionInfo", StringType(), True)
    ])
    
    # If input is a dictionary, convert to a Row
    if isinstance(input_data, dict):
        # Calculate age if not provided
        if 'Age' not in input_data and 'Year' in input_data:
            input_data['Age'] = 2025 - input_data['Year']
        
        # Add default values for missing fields
        for field in schema.fieldNames():
            if field not in input_data:
                if field == "PostedDate":
                    input_data[field] = "Current"
                elif field == "AdditionInfo":
                    input_data[field] = f"{input_data.get('Brand', '')} {input_data.get('model', '')}, {input_data.get('Year', '')}"
                else:
                    input_data[field] = None
        
        # Ensure proper types for fields
        processed_data = {}
        for field in schema.fields:
            field_name = field.name
            field_value = input_data.get(field_name)
            
            # Convert types as needed
            if field.dataType == IntegerType() and field_value is not None:
                try:
                    processed_data[field_name] = int(field_value)
                except (ValueError, TypeError):
                    processed_data[field_name] = None
            elif field.dataType == DoubleType() and field_value is not None:
                try:
                    processed_data[field_name] = float(field_value)
                except (ValueError, TypeError):
                    processed_data[field_name] = None
            else:
                processed_data[field_name] = field_value
        
        # Convert to Row
        row = Row(**processed_data)
        df = spark.createDataFrame([row], schema)
    
    # If already a DataFrame, ensure it has the right schema
    elif hasattr(input_data, 'columns'):
        # Convert to Spark DataFrame if it's a pandas DataFrame
        if not hasattr(input_data, 'rdd'):
            df = spark.createDataFrame(input_data)
        else:
            df = input_data
    
    else:
        raise ValueError("Input data must be a dictionary or DataFrame")
    
    return df

def predict_price(input_data):
    """Predict car price using the loaded model."""
    try:
        # Load models
        spark, model, feature_pipeline, metadata = load_models()
        
        # Format input data
        input_df = format_input_data(spark, input_data)
        
        # Apply feature pipeline
        processed_df = feature_pipeline.transform(input_df)
        
        # Make prediction
        prediction = model.transform(processed_df)
        
        # Extract predicted price
        predicted_price = prediction.select("prediction").first()[0]
        
        # Scale down predicted price by 15% to match ask prices better
        # (based on the observation that the model predicts higher than ask prices)
        adjusted_price = predicted_price * 0.85
        
        return {
            "original_prediction": predicted_price,
            "adjusted_prediction": adjusted_price,
            "model_info": metadata
        }
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def get_user_input():
    """Get user input for car features via CLI."""
    print("\n===== Used Car Price Predictor =====")
    
    features = {}
    
    # Get input for categorical features
    features['Brand'] = input("Enter car brand (e.g., Maruti): ")
    features['model'] = input("Enter car model (e.g., Swift): ")
    
    # Get input for FuelType with validation
    valid_fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
    while True:
        fuel_type = input(f"Enter fuel type {valid_fuel_types}: ")
        if fuel_type in valid_fuel_types:
            features['FuelType'] = fuel_type
            break
        else:
            print(f"Invalid fuel type. Please enter one of {valid_fuel_types}")
    
    # Get input for Transmission with validation
    valid_transmission = ['Manual', 'Automatic']
    while True:
        transmission = input(f"Enter transmission type {valid_transmission}: ")
        if transmission in valid_transmission:
            features['Transmission'] = transmission
            break
        else:
            print(f"Invalid transmission. Please enter one of {valid_transmission}")
    
    # Get input for Owner with validation
    valid_owners = ['first', 'second', 'third', 'fourth', 'Test Drive Car']
    while True:
        owner = input(f"Enter owner status {valid_owners}: ")
        if owner in valid_owners:
            features['Owner'] = owner
            break
        else:
            print(f"Invalid owner status. Please enter one of {valid_owners}")
    
    # Get input for numeric features with validation
    while True:
        try:
            year = int(input("Enter manufacturing year (e.g., 2018): "))
            if 1950 <= year <= 2025:
                features['Year'] = year
                break
            else:
                print("Year should be between 1950 and 2025")
        except ValueError:
            print("Please enter a valid year (numeric)")
    
    while True:
        try:
            km_driven = float(input("Enter kilometers driven: "))
            if km_driven >= 0:
                features['kmDriven'] = km_driven
                break
            else:
                print("Kilometers driven should be non-negative")
        except ValueError:
            print("Please enter a valid number for kilometers driven")
    
    # Calculate age from year
    features['Age'] = 2025 - features['Year']
    
    return features

def predict_price_cli():
    """Command-line interface for price prediction."""
    try:
        # Create spark session
        spark = create_spark_session()
        
        print("Loading models...")
        spark, model, feature_pipeline, metadata = load_models()
        print(f"Using {metadata['model_name']} model")
        
        while True:
            # Get user input
            input_data = get_user_input()
            
            # Format input data
            input_df = format_input_data(spark, input_data)
            
            # Make prediction
            print("\nPredicting price...")
            start_time = time.time()
            
            # Apply feature pipeline
            processed_df = feature_pipeline.transform(input_df)
            
            # Make prediction
            prediction = model.transform(processed_df)
            
            # Extract predicted price
            predicted_price = prediction.select("prediction").first()[0]
            
            # Scale down prediction by 15% to better match ask prices
            adjusted_price = predicted_price * 0.85
            
            prediction_time = time.time() - start_time
            
            # Display result
            print("\n===== Prediction Result =====")
            print(f"Estimated original price: ₹{predicted_price:,.2f}")
            print(f"Adjusted market price: ₹{adjusted_price:,.2f}")
            print(f"Prediction completed in {prediction_time:.4f} seconds")
            
            # Ask if user wants to make another prediction
            another = input("\nPredict another car price? (y/n): ").lower()
            if another != 'y':
                break
        
        # Stop Spark session
        spark.stop()
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict_price_cli()