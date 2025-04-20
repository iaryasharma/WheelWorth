# Used Car Price Estimation Tool

A machine learning application that predicts used car prices based on various features like brand, model, age, mileage, etc. This project uses Apache Spark's MLlib for building regression models and Streamlit for the user interface.

## Features

- **Data Processing**: Handles categorical variables, missing values, and feature engineering
- **Multiple ML Models**: Tests Linear Regression, Random Forest, and Gradient Boosting models
- **Model Evaluation**: Calculates RMSE, MSE, MAE, and R-squared metrics
- **Interactive UI**: User-friendly Streamlit interface for price predictions
- **Data Visualization**: Provides insights through various charts and graphs

## Project Structure

```
car-price-predictor/
├── data/
│   └── used_car_data.csv    # Your car dataset
├── src/
│   ├── train_model.py       # Model training script
│   ├── predict.py           # Prediction functionality
│   └── app.py               # Streamlit frontend
├── models/                  # Directory for saved models
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/car-price-predictor.git
cd car-price-predictor
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

Ensure your dataset (`data/used_car_data.csv`) has the following columns:
- Brand: Car manufacturer
- model: Car model
- Year: Manufacturing year
- Age: Age of the car in years
- kmDriven: Kilometers driven
- Transmission: Manual/Automatic
- Owner: Owner type (First Owner, Second Owner, etc.)
- FuelType: Type of fuel (Petrol, Diesel, etc.)
- PostedDate: Date when the car was listed
- AdditionInfo: Additional information about the car
- AskPrice: Price of the car (target variable)

## Usage

1. Train the machine learning model:
```bash
cd src
python train_model.py
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and go to the URL shown in the terminal (typically http://localhost:8501)

## Model Training

The training script:
- Preprocesses data (handling categorical variables, missing values)
- Trains multiple regression models (Linear Regression, Random Forest, Gradient Boosting)
- Evaluates and compares models using RMSE, MSE, and R-squared
- Saves the best-performing model

## Streamlit Interface

The Streamlit app provides:
- A form to input car details and get price estimates
- Data insights with visualizations
- Model performance metrics and charts

## Requirements

- Python 3.8+
- Apache Spark 3.4.0
- Streamlit 1.22.0
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.