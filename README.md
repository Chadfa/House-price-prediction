
# 🏠 Chennai House Price Predictor

This Streamlit-based machine learning app predicts housing prices in Chennai using features such as area, square footage, number of bedrooms, and number of bathrooms. It uses linear regression under the hood to make predictions and visualize model performance.

---

## 📁 Project Structure

```
.
├── house_price_app.py
├── Chennai housing sale.csv
└── README.md
```

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or manually install them:

```bash
pip install pandas numpy streamlit scikit-learn matplotlib seaborn
```

## 🚀 How to Run

1. Ensure you have Python 3.x installed.
2. Place the `Chennai housing sale.csv` file in the same folder as `house_price_app.py`.
3. In the terminal, navigate to the project folder.
4. Run the app using:

```bash
streamlit run house_price_app.py
```

5. Your browser will open automatically at `http://localhost:8501`.

---

## 🧠 Model Details

- **Algorithm**: Linear Regression
- **Preprocessing**: OneHotEncoding for categorical features (`AREA`)
- **Evaluation**: RMSE and R² Score

## 🖼️ Features

- Predict house prices by entering:
  - Area (location)
  - BHK (number of bedrooms)
  - Number of bathrooms
  - Square footage
- View model performance metrics (RMSE, R²)
- Explore the dataset through histograms and scatter plots

## 📊 Sample Screenshot

Check it out

## 📌 Notes

- Ensure the CSV file has the correct columns:
  - `AREA`, `INT_SQFT`, `N_BEDROOM`, `N_BATHROOM`, `SALES_PRICE`
- The app ignores rows with missing values.
