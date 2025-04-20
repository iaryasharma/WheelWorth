# pyspark_train_model.py
import os
import time
import json
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark_preprocessing import prepare_data

def train_models(train_df, test_df):
    """Train multiple regression models and return the best one."""
    # Initialize evaluator
    evaluator = RegressionEvaluator(
        labelCol="AskPrice", 
        predictionCol="prediction", 
        metricName="rmse"
    )
    
    # Define models to train
    models = {
        "Linear Regression": LinearRegression(featuresCol="scaled_features", labelCol="AskPrice"),
        "Random Forest": RandomForestRegressor(featuresCol="scaled_features", labelCol="AskPrice", numTrees=50),
        "Gradient Boosting": GBTRegressor(featuresCol="scaled_features", labelCol="AskPrice", maxIter=10)
    }
    
    best_model = None
    best_model_name = None
    best_rmse = float('inf')
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train the model
        trained_model = model.fit(train_df)
        
        # Make predictions on test data
        predictions = trained_model.transform(test_df)
        
        # Evaluate model
        rmse = evaluator.evaluate(predictions)
        r2 = evaluator.setMetricName("r2").evaluate(predictions)
        mae = evaluator.setMetricName("mae").evaluate(predictions)
        
        # Store results
        training_time = time.time() - start_time
        print(f"{name} training completed in {training_time:.2f} seconds")
        print(f"RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")
        
        results[name] = {
            "model": trained_model,
            "metrics": {
                "RMSE": rmse,
                "R2": r2,
                "MAE": mae
            }
        }
        
        # Track best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = trained_model
            best_model_name = name
    
    return results, best_model, best_model_name

def tune_best_model(train_df, test_df, best_model_name):
    """Tune hyperparameters for the best model."""
    print(f"\nTuning hyperparameters for {best_model_name}...")
    evaluator = RegressionEvaluator(
        labelCol="AskPrice", 
        predictionCol="prediction", 
        metricName="rmse"
    )
    
    if best_model_name == "Linear Regression":
        model = LinearRegression(featuresCol="scaled_features", labelCol="AskPrice")
        param_grid = ParamGridBuilder() \
            .addGrid(model.regParam, [0.01, 0.1, 0.5]) \
            .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()
    
    elif best_model_name == "Random Forest":
        model = RandomForestRegressor(featuresCol="scaled_features", labelCol="AskPrice")
        param_grid = ParamGridBuilder() \
            .addGrid(model.numTrees, [20, 50, 100]) \
            .addGrid(model.maxDepth, [5, 10, 15]) \
            .build()
    
    elif best_model_name == "Gradient Boosting":
        model = GBTRegressor(featuresCol="scaled_features", labelCol="AskPrice")
        param_grid = ParamGridBuilder() \
            .addGrid(model.maxDepth, [3, 5, 7]) \
            .addGrid(model.maxIter, [10, 20, 30]) \
            .build()
    
    else:
        print(f"No tuning parameters defined for {best_model_name}")
        return None
    
    # Create cross validator
    cv = CrossValidator(
        estimator=model,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3
    )
    
    # Train model with cross-validation
    cv_model = cv.fit(train_df)
    
    # Get the best model
    best_model = cv_model.bestModel
    
    # Make predictions
    predictions = best_model.transform(test_df)
    
    # Evaluate
    rmse = evaluator.evaluate(predictions)
    r2 = evaluator.setMetricName("r2").evaluate(predictions)
    mae = evaluator.setMetricName("mae").evaluate(predictions)
    
    print(f"Tuned {best_model_name} performance:")
    print(f"RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")
    
    return {
        "model": best_model,
        "metrics": {
            "RMSE": rmse,
            "R2": r2, 
            "MAE": mae
        }
    }

def save_model(model, feature_pipeline, model_name, metrics, original_df, output_dir='models'):
    """Save the trained model and related artifacts."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, "car_price_model")
    model.save(model_path)
    
    # Save feature pipeline
    pipeline_path = os.path.join(output_dir, "feature_pipeline")
    feature_pipeline.save(pipeline_path)
    
    # Save metadata (model name and metrics)
    metadata = {
        "model_name": model_name,
        "metrics": {
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "MAE": metrics["MAE"]
        },
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Save feature importance if applicable
    if hasattr(model, "featureImportances"):
        feature_importance = model.featureImportances.toArray().tolist()
        with open(os.path.join(output_dir, "feature_importance.json"), 'w') as f:
            json.dump(feature_importance, f, indent=4)
    
    # Save sample data for the app
    sample_data_path = os.path.join(output_dir, "sample_data.csv")
    original_pandas_df = original_df.toPandas()
    original_pandas_df.to_csv(sample_data_path, index=False)
    
    print(f"\nModel saved to {model_path}")
    print(f"Feature pipeline saved to {pipeline_path}")
    print(f"Model metadata saved to {metadata_path}")
    print(f"Sample data saved to {sample_data_path}")
    
    return {
        "model_path": model_path,
        "pipeline_path": pipeline_path,
        "metadata_path": metadata_path
    }

def main(data_path):
    """Main function to train and evaluate models."""
    print(f"Loading data from {data_path}...")
    
    # Prepare data
    train_df, test_df, feature_pipeline, spark, original_df = prepare_data(data_path)
    
    # Train models
    print("\nTraining models...")
    model_results, best_model, best_model_name = train_models(train_df, test_df)
    
    # Tune the best model
    tuned_result = tune_best_model(train_df, test_df, best_model_name)
    
    # Use the tuned model if it improves results
    if tuned_result and tuned_result["metrics"]["RMSE"] < model_results[best_model_name]["metrics"]["RMSE"]:
        print(f"\nTuned {best_model_name} performs better. Using the tuned model.")
        final_model = tuned_result["model"]
        final_metrics = tuned_result["metrics"]
    else:
        print(f"\nOriginal {best_model_name} performs better. Using the original model.")
        final_model = model_results[best_model_name]["model"]
        final_metrics = model_results[best_model_name]["metrics"]
    
    # Save the model and related artifacts
    saved_paths = save_model(
        final_model, 
        feature_pipeline, 
        best_model_name, 
        final_metrics,
        original_df
    )
    
    # Stop Spark session
    spark.stop()
    
    return {
        "best_model_name": best_model_name,
        "metrics": final_metrics,
        "saved_paths": saved_paths
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        result = main(data_path)
        print("\nModel training completed successfully!")
    else:
        print("Please provide a data path as argument")