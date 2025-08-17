#!/usr/bin/env python3
"""
House Price Prediction Script
This script loads a trained model and makes predictions on new house data.
"""

import pandas as pd
import sys
from model import HousePricePredictor

def get_user_input():
    """Get house features from user input"""
    print("\n🏠 Enter house details for prediction:")
    print("-" * 40)
    
    try:
        square_feet = float(input("Square feet: "))
        bedrooms = int(input("Number of bedrooms: "))
        bathrooms = int(input("Number of bathrooms: "))
        garage = int(input("Garage spaces: "))
        age = int(input("House age (years): "))
        neighborhood = input("Neighborhood (suburban/urban/rural): ").lower()
        
        return pd.DataFrame({
            'square_feet': [square_feet],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'garage': [garage],
            'age': [age],
            'neighborhood': [neighborhood]
        })
        
    except ValueError as e:
        print(f" Invalid input: {e}")
        return None

def main():
    """Main prediction function"""
    print(" House Price Prediction")
    print("=" * 30)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Check if model exists
    model_path = 'house_price_model.pkl'
    if not pd.io.common.file_exists(model_path):
        print(" Model not found. Please train the model first using train.py")
        sys.exit(1)
    
    # Load trained model
    try:
        predictor.load_model(model_path)
        print(" Model loaded successfully!")
    except Exception as e:
        print(f" Error loading model: {e}")
        sys.exit(1)
    
    # Get user input
    house_data = get_user_input()
    if house_data is None:
        sys.exit(1)
    
    # Make prediction
    try:
        predicted_price = predictor.predict(house_data)
        print(f"\n Predicted house price: ${predicted_price[0]:,.2f}")
        
    except Exception as e:
        print(f" Error making prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
