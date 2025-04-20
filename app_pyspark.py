import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from pyspark.sql import SparkSession
from pyspark_predict import predict_price, format_input_data, create_spark_session, load_models

# Set page configuration
st.set_page_config(
    page_title="WheelWorth - Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

# Initialize Spark session at startup
@st.cache_resource
def get_spark():
    return create_spark_session("Car Predictor Web App")

def load_data():
    """Load the sample data for exploration."""
    # Try to load saved data first
    sample_data_path = "models/sample_data.csv"
    if os.path.exists(sample_data_path):
        try:
            df = pd.read_csv(sample_data_path)
            return df
        except Exception as e:
            st.error(f"Error loading saved data: {e}")
    
    st.error(f"Sample data not found at {sample_data_path}. Please train a model first.")
    return None

def load_model_info():
    """Load model information from metadata file."""
    metadata_path = "models/model_metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading model metadata: {e}")
    else:
        st.warning("No model metadata found. Please train a model first.")
    
    return {
        "model_name": "Unknown",
        "metrics": {
            "RMSE": 0.0,
            "R2": 0.0,
            "MAE": 0.0
        },
        "creation_date": "Unknown"
    }

def make_prediction(input_features):
    """Make a prediction using the loaded model."""
    try:
        # Get spark session
        spark = get_spark()
        
        # Make prediction
        start_time = time.time()
        result = predict_price(input_features)
        prediction_time = time.time() - start_time
        
        return {
            "original_price": result["original_prediction"],
            "adjusted_price": result["adjusted_prediction"],
            "prediction_time": prediction_time
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def run_data_exploration(df):
    """Run data exploration on the sample data."""
    st.subheader("Dataset Overview")
    
    # Show basic statistics
    st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display sample data
    with st.expander("View Sample Data"):
        st.dataframe(df.head(10))
    
    # Display descriptive statistics
    with st.expander("View Descriptive Statistics"):
        st.dataframe(df.describe())
    
    # Create distribution plots
    st.subheader("Price Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['AskPrice'], kde=True, ax=ax)
        ax.set_title('Car Price Distribution')
        ax.set_xlabel('Price (₹)')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(y=df['AskPrice'], ax=ax)
        ax.set_title('Car Price Boxplot')
        ax.set_ylabel('Price (₹)')
        st.pyplot(fig)
    
    # Relationship between price and other numeric variables
    st.subheader("Price Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Year', y='AskPrice', data=df, alpha=0.6, ax=ax)
        ax.set_title('Price vs. Year')
        ax.set_xlabel('Manufacturing Year')
        ax.set_ylabel('Price (₹)')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='kmDriven', y='AskPrice', data=df, alpha=0.6, ax=ax)
        ax.set_title('Price vs. Kilometers Driven')
        ax.set_xlabel('Kilometers Driven')
        ax.set_ylabel('Price (₹)')
        st.pyplot(fig)
    
    # Price by categorical variables
    st.subheader("Price by Categories")
    
    categorical_cols = ['Brand', 'FuelType', 'Transmission', 'Owner']
    cat_cols_available = [col for col in categorical_cols if col in df.columns]
    
    selected_cat = st.selectbox("Select category:", cat_cols_available)
    
    if selected_cat:
        # Get top 10 categories by count
        top_categories = df[selected_cat].value_counts().nlargest(10).index
        filtered_df = df[df[selected_cat].isin(top_categories)]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x=selected_cat, y='AskPrice', data=filtered_df, ax=ax)
        ax.set_title(f'Price by {selected_cat}')
        ax.set_ylabel('Price (₹)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

def main():
    # Initialize spark if not already done
    spark = get_spark()
    
    # Load model metadata
    model_info = load_model_info()
    
    # App header
    st.title("🚗 WheelWorth - Used Car Price Predictor")
    st.markdown("""
    Predict the price of your used car based on its features using machine learning.
    This app uses a trained PySpark ML model to estimate market values.
    """)
    
    # Model info section
    st.sidebar.header("Model Information")
    st.sidebar.write(f"**Model Type:** {model_info['model_name']}")
    st.sidebar.write(f"**Created on:** {model_info.get('creation_date', 'Unknown')}")
    st.sidebar.write(f"**Model Performance:**")
    st.sidebar.write(f"- R² Score: {model_info['metrics'].get('R2', 0):.4f}")
    st.sidebar.write(f"- RMSE: ₹{model_info['metrics'].get('RMSE', 0):,.2f}")
    st.sidebar.write(f"- MAE: ₹{model_info['metrics'].get('MAE', 0):,.2f}")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Make Prediction", "Data Exploration"])
    
    with tab1:
        st.header("Predict Car Price")
        
        # Create columns for inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brand = st.text_input("Car Brand", value="Maruti")
            model = st.text_input("Car Model", value="Swift")
            year = st.number_input("Manufacturing Year", min_value=1950, max_value=2025, value=2018)
            
        with col2:
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
            owner = st.selectbox("Owner", ["first", "second", "third", "fourth", "Test Drive Car"])
            
        with col3:
            km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
            # Calculate age from year
            age = 2025 - year
            st.write(f"Car Age: {age} years")
        
        # Create input features dictionary
        input_features = {
            'Brand': brand,
            'model': model,
            'Year': year,
            'Age': age,
            'kmDriven': km_driven,
            'Transmission': transmission,
            'Owner': owner,
            'FuelType': fuel_type,
            'PostedDate': 'Current',
            'AdditionInfo': f"{brand} {model}, {year}"
        }
        
        # Prediction button
        if st.button("Predict Price"):
            with st.spinner("Calculating price..."):
                # Make prediction
                result = make_prediction(input_features)
                
                if result:
                    # Display results
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Estimated Market Price", 
                            value=f"₹{result['adjusted_price']:,.2f}"
                        )
                        st.caption("This is the recommended asking price (adjusted for market conditions)")
                    
                    with col2:
                        st.metric(
                            label="Original Prediction", 
                            value=f"₹{result['original_price']:,.2f}"
                        )
                        st.caption("Raw model prediction before market adjustment")
                    
                    st.info(f"Prediction completed in {result['prediction_time']:.4f} seconds")
                    
                    # Display reasoning
                    st.subheader("Price Factors")
                    st.markdown("""
                    The predicted price is based on these key factors:
                    - **Car age and year**: Newer cars typically have higher values
                    - **Kilometers driven**: Higher mileage decreases value
                    - **Brand and model**: Premium brands retain value better
                    - **Fuel type**: Petrol vs diesel affects resale value
                    - **Transmission**: Automatic typically commands higher prices
                    - **Ownership history**: First owner vehicles have better value
                    """)
    
    with tab2:
        st.header("Data Exploration")
        
        # Load sample data
        df = load_data()
        
        if df is not None:
            run_data_exploration(df)
        else:
            st.warning("Data exploration is not available. Please train a model first to generate sample data.")

if __name__ == "__main__":
    main()