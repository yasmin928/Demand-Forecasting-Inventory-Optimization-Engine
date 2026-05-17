# 📦 Demand Forecasting & Inventory Optimization Engine

🚀 An end-to-end machine learning system that predicts product demand and optimizes inventory decisions using LightGBM, XGBoost, MLflow tracking, and a Flask analytics dashboard.

## 🧠 Business Problem

Retail businesses constantly struggle with:
- 📉 Stockouts → Lost revenue + unhappy customers  
- 📈 Overstock → Increased storage + capital costs  

🎯 Objective: Build an intelligent system that forecasts demand, identifies inventory risks, and supports data-driven replenishment decisions.

## 🏗️ System Architecture

Raw Data → Data Cleaning & Feature Engineering → Time Series Feature Creation → Model Training (LightGBM / XGBoost) → MLflow Tracking → Flask API → Interactive Dashboard (Chart.js)

## 📊 Dataset Overview

Retail inventory dataset includes:
- Store ID, Product ID  
- Inventory Level, Units Sold, Units Ordered  
- Pricing & Discount data  
- Competitor pricing  
- Weather conditions  
- Region & seasonality  
- Holiday & promotion flags  

## ⚙️ Feature Engineering

### 🔹 Data Preprocessing
- Missing value handling  
- Data type standardization  
- Encoding (Label / One-Hot Encoding)  

### 🔹 Time-Series Features
- Lag features: lag_1, lag_7, lag_14, lag_30  
- Rolling mean & standard deviation  
- Trend & seasonality extraction  
- Stationarity testing (ADF test)  

## 🤖 Machine Learning Models

- ⚡ LightGBM Regressor  
- ⚡ XGBoost Regressor  

Models predict:
👉 Future Units Sold (Demand Forecasting)

## 📦 Inventory Optimization Logic

- 🔴 Stockout Risk → Inventory < 0.5 × Forecasted Demand  
- 🟡 Overstock Risk → Inventory > 1.5 × Forecasted Demand  
- 🟢 Normal → Otherwise  

## 📈 Dashboard Features

Built using Flask + Chart.js:

### 📊 Core Analytics
- Inventory status overview  
- Demand vs inventory comparison  
- Stock trend analysis  

### 🔍 Filters
- Store  
- Product  
- Category  
- Region  

### 📉 Visualizations
- Status distribution (donut chart)  
- Inventory vs sales comparison  
- Overstock ranking  
- Stockout-risk ranking  
- Category breakdown  

## 🧪 MLflow Tracking

MLflow is used to track:
- Model versions  
- Metrics (MAE, RMSE, R²)  
- Experiments for LightGBM and XGBoost  
- Feature configurations  

## 📁 Project Structure

demand-forecasting/
├── data/
├── notebooks/
├── models/
├── mlruns/
├── templates/
│   └── dashboard.html
├── app.py
├── train.py
└── README.md

## 🚀 How to Run

### Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost flask mlflow matplotlib seaborn

### Run MLflow UI (optional)
mlflow ui

### Run Flask app
python app.py

### Open dashboard
http://127.0.0.1:5000

## 📊 Key Insights

- Demand shows strong seasonal patterns  
- Some SKUs consistently cause overstock  
- Others frequently hit stockout risk  
- ML significantly improves inventory visibility and decision-making  

## 🔮 Future Improvements

- Real-time forecasting pipeline  
- Auto-replenishment system  
- Cloud deployment (AWS / Azure / GCP)  
- Full MLOps CI/CD pipeline  
- Advanced anomaly detection for demand spikes  
