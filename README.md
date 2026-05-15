# 📦 Demand Forecasting & Inventory Optimization Engine

🚀 An end-to-end Data Science project that predicts product demand and optimizes inventory decisions using Machine Learning and an interactive dashboard.

## 🧠 Business Problem

Retail companies often face:
- 📉 Stockouts → Lost sales & customer dissatisfaction  
- 📈 Overstock → High holding & storage costs  

🎯 Goal: Build a smart system to forecast demand and optimize inventory decisions.

## 🏗️ Project Architecture

Raw Data → Cleaning → Feature Engineering → Model Training → API → Dashboard

## 📊 Dataset

Retail Store Inventory Dataset including:
- Store ID, Product ID  
- Inventory Level, Units Sold, Units Ordered  
- Price, Discount, Competitor Pricing  
- Weather Condition, Region, Seasonality  
- Holiday / Promotion  

## ⚙️ Key Features

### 🔹 Data Processing
- Missing value handling  
- Data type fixing  
- Feature engineering (lag & rolling features)  
- Encoding (Label Encoding + One-Hot Encoding)

### 🔹 Time Series Engineering
- Lag Features: lag_1, lag_7, lag_14, lag_30  
- Rolling mean & standard deviation  
- Stationarity Check (ADF Test)

### 🔹 Machine Learning Models
- LightGBM  
- XGBoost  

### 🔹 Inventory Optimization Logic

Inventory > 1.5 × Demand → 🟡 Overstock  
Inventory < 0.5 × Demand → 🔴 Stockout  
Otherwise → 🟢 Normal  

## 📈 Dashboard Features

Built using Flask + HTML + Chart.js:
- 📊 Inventory Status Summary  
- 🥧 Pie Chart  
- 🔍 Filters: Store, Product, Category, Region  
- 📋 Product-level status table  

## 🧪 Model Performance

XGBoost → MAE ~69 | RMSE ~89 | R² ~0.33  
LightGBM → MAE ~69 | RMSE ~89 | R² ~0.33  

## 📁 Project Structure

demand-forecasting/  
├── data/  
├── notebooks/  
├── models/  
├── templates/  
├── app.py  
└── README.md  

## 🚀 How to Run

Install dependencies: pip install pandas numpy scikit-learn lightgbm xgboost flask matplotlib seaborn  

Run the app: python app.py  

Open in browser: http://127.0.0.1:5000  

## 📊 Key Insights

- Demand shows seasonal patterns  
- Some products are consistently overstocked  
- Others frequently face stockouts  
- Inventory decisions can be improved using ML  

## 🔮 Future Improvements

- Real-time forecasting API  
- Auto reorder recommendation system  
- Cloud deployment (Azure / AWS)  
- MLOps pipeline  

