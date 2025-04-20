import argparse
import os
import sys
from pyspark_train_model import main as train_main
from pyspark_predict import predict_price_cli, load_models

def train_model(data_path):
    """Train a model using the specified data."""
    try:
        print(f"Training PySpark model using data from: {data_path}")
        
        # Verify data file exists
        if not os.path.exists(data_path):
            print(f"Error: Data file not found at {data_path}")
            return False
        
        # Train model
        result = train_main(data_path)
        print(f"Model training completed successfully. Best model: {result['best_model_name']}")
        print(f"Model metrics: RMSE={result['metrics']['RMSE']:.2f}, R²={result['metrics']['R2']:.4f}")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_models_exist():
    """Verify that trained models exist."""
    try:
        # Try to load models to verify they exist and are working
        _, _, _, _ = load_models()
        return True
    except Exception as e:
        print(f"Error verifying models: {str(e)}")
        return False

def run_prediction():
    """Run the prediction interface."""
    try:
        # Verify model exists before starting prediction
        if not verify_models_exist():
            print("Error: Model files not found or invalid. Please train the model first.")
            return False
            
        predict_price_cli()
        return True
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Used Car Price Predictor (PySpark version)')
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='predict',
                      help='Mode to run: train a new model, make predictions, or both')
    parser.add_argument('--data', type=str, default='data/used_car_data.csv',
                      help='Path to the training data CSV file')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        success = train_model(args.data)
        if not success:
            print("Model training failed.")
            if args.mode == 'both':
                print("Skipping prediction step.")
                return
    
    if args.mode in ['predict', 'both']:
        success = run_prediction()
        if not success:
            print("Prediction interface failed.")

if __name__ == "__main__":
    main()