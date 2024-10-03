import streamlit as st
import pandas as pd
import pickle

# Load the trained Lasso regression model
filename = 'lasso_regression_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a Streamlit app
st.title("Monthly Revenue Prediction App")

# Input features (replace with your actual feature names)
st.sidebar.header("Input Features")
total_orders = st.sidebar.number_input("Total Orders")
average_order_value = st.sidebar.number_input("Average Order Value")
customer_acquisition_cost = st.sidebar.number_input("Customer Acquisition Cost")
# Add more input fields for other features as needed

# Create a dictionary with the input values
input_data = {
    'total_orders': total_orders,
    'average_order_value': average_order_value,
    'customer_acquisition_cost': customer_acquisition_cost,
    # Add more features as needed
}

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Make predictions using the loaded model
if st.sidebar.button("Predict"):
    prediction = loaded_model.predict(input_df)
    st.write("Predicted Monthly Revenue:", prediction[0])

# You can add more elements to your app, such as charts, explanations, etc.
