# 🏠 House Price Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📌 Project Overview

This project builds an end-to-end **Machine Learning Regression Pipeline** to predict house prices using property-related features such as bedrooms, bathrooms, square footage, and location.

The project demonstrates the complete ML workflow:

- ✔ Exploratory Data Analysis (EDA)
- ✔ Data Cleaning
- ✔ Feature Engineering
- ✔ Outlier Removal (IQR Method)
- ✔ Model Training & Comparison
- ✔ Model Evaluation
- ✔ Model Saving for Deployment

Two regression models were implemented and compared:

- 🌳 Random Forest Regressor
- 📈 Linear Regression

---

## 🎯 Problem Statement

Predict house prices accurately based on property features to assist:

- Real estate agencies
- Property buyers and sellers
- Investment analysts
- Financial institutions

The target variable is:

```python
price
```

---

## 📊 Dataset Description

The dataset contains housing features such as:

| Feature | Description |
|----------|--------------|
| price | Target variable |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| sqft_living | Living area (sq ft) |
| sqft_lot | Lot size |
| floors | Number of floors |
| waterfront | Waterfront property (0/1) |
| condition | Condition rating |
| view | View rating |
| yr_built | Year built |
| yr_renovated | Year renovated |
| city | Property location |
| date | Sale date |

---

## 🔎 Exploratory Data Analysis (EDA)

Performed to understand:

- Dataset structure
- Missing values
- Feature distribution
- Correlation between features
- Outliers

### Visualizations Used

- Scatter plots (Price vs Living Area, Bedrooms, Bathrooms)
- Count plots
- Box plots
- Correlation heatmap

### Key Insights

- `sqft_living` has strong correlation with price.
- Some extreme outliers were present.
- Categorical imbalance in city feature.

---

## 🛠 Data Preprocessing

### 1️⃣ Date Feature Engineering
- Converted `date` to datetime format
- Extracted:
  - Year
  - Month
  - Day

---

### 2️⃣ Handling Missing Values
- Numerical columns → filled with **median**
- Categorical columns → filled with **mode**
- Removed duplicate rows

---

### 3️⃣ Outlier Removal (IQR Method)

Used Interquartile Range:

```
IQR = Q3 - Q1
```

Removed values outside:

```
Q1 - 1.5 * IQR
Q3 + 1.5 * IQR
```

Applied on:
- price
- sqft_living
- sqft_lot

---

### 4️⃣ Feature Engineering

- Created new feature:
  ```
  is_renovated
  ```
- Removed unnecessary columns:
  - street
  - country
  - statezip
  - yr_built
- Grouped rare cities into `"Other"`
- Applied One-Hot Encoding

---

### 5️⃣ Feature Scaling

Used `StandardScaler` to standardize features before training.

---

## 🤖 Model Training

### 🌳 Random Forest Regressor
- 300 decision trees
- Handles non-linear relationships
- Reduces overfitting
- Captures feature interactions

### 📈 Linear Regression
- Baseline model
- Assumes linear relationship
- Simple and interpretable

---

## 📊 Model Evaluation Metrics

| Metric | Description |
|----------|--------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² Score | Variance explained by model |

---

## 🏆 Results

- Random Forest outperformed Linear Regression.
- Achieved higher R² score.
- Lower RMSE compared to Linear Regression.
- Better handling of complex and non-linear patterns.

---

## 💾 Model Saving

Saved using `joblib`:

- `house_price_model.pkl`
- `linear_regression_model.pkl`
- `scaler.pkl`
- `feature_names.pkl`

These files allow easy deployment in web applications such as Streamlit.

---

## 📂 Project Structure

```
├── data.csv
├── processed_house_data.csv
├── house_price_model.pkl
├── linear_regression_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── house_price_model.py
├── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Install Dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3️⃣ Run Script

```
python house_price_model.py
```

---

## 🧠 Key Learnings

- Importance of proper EDA
- Handling missing values effectively
- Removing outliers improves performance
- Feature engineering boosts model accuracy
- Random Forest is powerful for real-world regression
- Saving models enables deployment

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Cross-validation
- Feature importance visualization
- Model explainability (SHAP)
- Streamlit deployment
- CI/CD integration

---

## 👨‍💻 Author

**Kais Anjum**  
Machine Learning Enthusiast  


---

⭐ If you found this project useful, feel free to star the repository!
