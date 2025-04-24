import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Chennai housing sale.csv")

# Clean data
key_columns = ['AREA', 'INT_SQFT', 'N_BEDROOM', 'N_BATHROOM', 'SALES_PRICE']
df_clean = df.dropna(subset=key_columns).copy()
df_clean.reset_index(drop=True, inplace=True)

# Features and target
X = df_clean[['AREA', 'INT_SQFT', 'N_BEDROOM', 'N_BATHROOM']]
y = df_clean['SALES_PRICE']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = ['AREA']
numerical_features = ['INT_SQFT', 'N_BEDROOM', 'N_BATHROOM']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Pipeline with linear regression
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Prediction function
def predict_price(area, sqft, bhk, bath):
    input_df = pd.DataFrame([[area, sqft, bhk, bath]], columns=['AREA', 'INT_SQFT', 'N_BEDROOM', 'N_BATHROOM'])
    return model.predict(input_df)[0]

# Streamlit UI
st.title("🏠 Chennai House Price Predictor")

location = st.selectbox("Select the Location:", df_clean['AREA'].unique()) 
bhk = st.number_input("Enter BHK:", min_value=1, max_value=10, step=1) 
bath = st.number_input("Enter Number of Bathrooms:", min_value=1, max_value=10, step=1)
sqft = st.number_input("Enter Square Feet:", min_value=100, max_value=10000, step=50)

if st.button("Predict Price"):
    result = predict_price(location, sqft, bhk, bath)
    st.success(f"💰 Predicted Price: ₹ {result:,.2f}")

# Model performance section
with st.expander("📊 Model Performance"):
    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**R² Score**: {r2:.4f}")

# Data exploration section
with st.expander("📂 Explore Data"):
    st.write("Sample of Data:")
    st.dataframe(df_clean.head())

    fig1, ax1 = plt.subplots()
    sns.histplot(df_clean['SALES_PRICE'], kde=True, ax=ax1)
    ax1.set_title("Distribution of Sales Price")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df_clean, x='INT_SQFT', y='SALES_PRICE', hue='AREA', ax=ax2)
    ax2.set_title("Price vs Square Feet")
    st.pyplot(fig2)
