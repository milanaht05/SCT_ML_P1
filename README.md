#  House Price Prediction Project

A machine learning project that predicts house prices based on various features like square footage, number of bedrooms, bathrooms, and location.

##  Project Structure

```
house_price_prediction/
├── dataset/
│   └── house_data.csv          # Sample dataset with 20 house listings
├── model.py                    # Core prediction model class
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

##  Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

### 3. Make Predictions
```bash
python predict.py
```

##  Dataset Features

The dataset (`house_data.csv`) contains the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `square_feet` | int | Total living area in square feet |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | int | Number of bathrooms |
| `garage` | int | Number of garage spaces |
| `age` | int | Age of the house in years |
| `neighborhood` | str | Location type (suburban/urban/rural) |
| `price` | int | House price in USD (target variable) |

##  Model Details

- **Algorithm**: Linear Regression
- **Preprocessing**: One-hot encoding for categorical features
- **Metrics**: RMSE, R² Score
- **Model File**: `house_price_model.pkl`

##  Example Usage

### Training Output:
```
 House Price Prediction Model Training
==================================================
 Loading dataset...
 Dataset loaded: 20 samples

 Dataset preview:
   square_feet  bedrooms  bathrooms  garage  age neighborhood   price
0         1500         3          2       1   10     suburban  300000
1         2000         4          3       2    5     suburban  400000
...

 Model trained successfully!
   RMSE: $45,234.56
   R² Score: 0.8923

 Model saved to house_price_model.pkl
```

### Prediction Output:
```
 House Price Prediction
==============================
Enter house details for prediction:
----------------------------------------
Square feet: 1800
Number of bedrooms: 3
Number of bathrooms: 2
Garage spaces: 1
House age (years): 5
Neighborhood (suburban/urban/rural): suburban

 Predicted house price: $350,000.00
```

##  How to Use

### For Developers:
1. **Training**: Run `python train.py` to train a new model
2. **Prediction**: Run `python predict.py` for interactive predictions
3. **Custom Data**: Modify `dataset/house_data.csv` with your own data

### For Integration:
```python
from model import HousePricePredictor

# Load trained model
predictor = HousePricePredictor()
predictor.load_model('house_price_model.pkl')

# Make predictions
new_house = pd.DataFrame({
    'square_feet': [2000],
    'bedrooms': [4],
    'bathrooms': [3],
    'garage': [2],
    'age': [3],
    'neighborhood': ['suburban']
})

price = predictor.predict(new_house)
print(f"Predicted price: ${price[0]:,.2f}")
```

##  Technical Stack

- **Python 3.7+**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **numpy** - Numerical computations
- **joblib** - Model persistence

##  Notes

- The dataset is a sample with 20 entries for demonstration purposes
- For production use, consider using a larger dataset
- You can enhance the model by trying different algorithms (Random Forest, XGBoost, etc.)
- Feature engineering can improve predictions (add features like lot size, school ratings, etc.)

##  Contributing

Feel free to fork this project and submit pull requests for improvements!
