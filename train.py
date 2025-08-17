#!/usr/bin/env python3
"""
House Price Prediction Training Script
This script trains a house price prediction model using the provided dataset.
"""

import os
import sys
from model import HousePricePredictor

def main():
    """Main training function"""
    print(" House Price Prediction Model Training")
    print("=" * 50)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Check if dataset exists
    dataset_path = 'dataset/house_data.csv'
    if not os.path.exists(dataset_path):
        print(f" Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Load and train model
    try:
        print(" Loading dataset...")
        df = predictor.load_data(dataset_path)
        print(f" Dataset loaded: {len(df)} samples")
        
        print("\n Dataset preview:")
        print(df.head())
        print("\n Dataset statistics:")
        print(df.describe())
        
        # Preprocess data
        print("\n Preprocessing data...")
        X, y, preprocessor = predictor.preprocess_data(df)
        
        # Train model
        print("\n Training model...")
        metrics = predictor.train_model(X, y, preprocessor)
        
        print("\n Model training completed!")
        print(f"   RMSE: ${metrics['rmse']:,.2f}")
        print(f"   R² Score: {metrics['r2']:.4f}")
        print(f"   Mean Absolute Error: ${np.sqrt(metrics['mse']):,.2f}")
        
        # Save model
        model_path = 'house_price_model.pkl'
        predictor.save_model(model_path)
        print(f"\n Model saved to {model_path}")
        
        print("\n Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import numpy as np
    main()
