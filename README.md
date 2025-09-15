# 💹 Stock Price Predictor

A simple Python-based local web app to predict stock prices using LSTM neural networks.  
Users can enter a stock symbol or company name, view historical trends, and get future price predictions.

---

## 🛠️ Tech Stack
- **Backend & Web:** Flask, Python  
- **Data & ML:** Pandas, NumPy, TensorFlow, Keras, yfinance, yahooquery  
- **Visualization:** Matplotlib  

---

## 🚀 Features
- 🔍 Enter stock symbol (e.g., `AAPL`) or company name (e.g., `Apple`)  
- 📈 Predict future stock prices using an LSTM model  
- 📊 Visualize historical trends with matplotlib charts  
- 💾 Works locally — no server deployment required  

---

## 📄 Model Overview
- LSTM neural network trained on historical stock prices  
- Generates predictions for a user-specified number of future days  
- Fetches stock data in real-time using Yahoo Finance  
- Accepts company names and converts them to stock symbols using `yahooquery`  

---

## ⚙️ Getting Started

### 1️⃣ Clone & Install
```bash
git clone https://github.com/SreeragSreekanth/stockprice-predictor.git
cd stockprice-predictor
python -m venv venv

# Activate the virtual environment
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2️⃣ Run the App
```bash
python app.py
```

Open your browser and go to:
```
http://127.0.0.1:5000
```

🧾 Usage

Enter a stock symbol or company name.

Select the number of days to predict.

View historical stock chart and future predictions.

⚠️ Notes

Predictions are based on historical stock data and cannot guarantee future prices.

Company names are resolved to symbols using yahooquery. If a company is missing, the symbol might not be found.

✨ Author

Built with ❤️ by [Sreerag Sreekanth](https://github.com/SreeragSreekanth)
