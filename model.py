import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        
    def load_data(self, filepath):
        """Load the house price dataset"""
        return pd.read_csv(filepath)
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        X = df.drop('price', axis=1)
        y = df['price']
        
        # Define categorical and numerical features
        categorical_features = ['neighborhood']
        numerical_features = ['square_feet', 'bedrooms', 'bathrooms', 'garage', 'age']
        
        # Create preprocessing pipelines
        categorical_transformer = OneHotEncoder(drop='first')
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', 'passthrough', numerical_features)
            ])
        
        return X, y, preprocessor
    
    def train_model(self, X, y, preprocessor):
        """Train the house price prediction model"""
        # Create pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }
    
    def predict(self, features):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        return self.model.predict(features)
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = joblib.load(filepath)
