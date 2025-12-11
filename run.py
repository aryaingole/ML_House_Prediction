# run.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import pickle 

# IMPORTANT: Import the function from the second file
from html_content import generate_html_report 

# --- 1. Data Preparation and Model Training ---

def train_and_save_model():
    """Fetches data, trains the model, and saves it to a file."""
    
    print("Loading California Housing Dataset...")
    housing = fetch_california_housing(as_frame=True)
    
    # We use four key features: Median Income, Avg Rooms, Latitude, and Longitude
    X = housing.data[['MedInc', 'AveRooms', 'Latitude', 'Longitude']]
    y = housing.target # Median House Value (the target)
    
    # Split data chronologically (or randomly with random_state=42 for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and Train the Linear Regression Model
    model = LinearRegression()
    print("Training Linear Regression model...")
    model.fit(X_train, y_train)
    
    # Save the trained model to a file
    model_filename = 'house_price_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved as {model_filename}")
    
    score = model.score(X_test, y_test)
    print(f"Model Test Score (R-squared): {score:.4f}")
    
    return model_filename

# --- 2. Prediction Function ---

def predict_house_price(model_filename, input_features):
    """Loads the trained model and makes a prediction."""
    
    try:
        # Load the saved model
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return "Error: Model file not found. Please run train_and_save_model() first."

    # Reshape the input features for the model (expects 2D array)
    data = np.array(input_features).reshape(1, -1)
    
    prediction = model.predict(data)
    
    # Convert the prediction from $100,000s to actual dollars
    predicted_price = prediction[0] * 100000 
    
    return predicted_price

# --- 3. Main Execution Block ---

if __name__ == '__main__':
    
    # 1. Check for and load/train the model
    model_file = 'house_price_model.pkl'
    try:
        with open(model_file, 'rb') as f:
            print("Trained model found. Skipping training.")
    except FileNotFoundError:
        model_file = train_and_save_model()
        
    # 2. Define the input features for the prediction:
    # Example Inputs: [Median Income (in $10k), Avg Rooms, Latitude, Longitude]
    example_input = [5.0, 6.0, 34.0, -118.0] 
    final_price = predict_house_price(model_file, example_input)
    
    print("\n--- Example Prediction ---")
    print(f"Input Features: {example_input}")
    print(f"Predicted Median House Price: ${final_price:,.2f}")
    
    # 3. Generate and Save the HTML Report
    html_output = generate_html_report(example_input, final_price)
    
    report_filename = 'prediction_report.html'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_output)
        
    print(f"\n✅ SUCCESS! Prediction report saved to: {report_filename}")
    print("Open this file in your web browser to view the result.")