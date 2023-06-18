import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
model = pickle.load(open('dt_classifier.pkl', 'rb'))

# Function to preprocess input data
def preprocess_input(data):
    # Perform any necessary preprocessing steps here
    # Example: One-hot encoding, feature scaling, etc.
    return data

# Function to predict car price
def predict_price(car_data):
    preprocessed_data = preprocess_input(car_data)
    predicted_price = model.predict(preprocessed_data)
    return predicted_price

# Set page title
st.title('Car Price Prediction')

# Create input fields for car details
st.subheader('Enter Car Details')
name = st.text_input('Name')
year = st.number_input('Year')
km_driven = st.number_input('Kilometers Driven')
fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

# Create a button to predict car price
if st.button('Predict Price'):
    # Create a dictionary to hold the car details
    car_data = {
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    }

    # Convert the dictionary into a DataFrame
    car_df = pd.DataFrame(car_data)

    # Predict the car price
    predicted_price = predict_price(car_df)

    # Display the predicted price
    st.subheader('Predicted Car Price')
    st.write(f"The predicted price of the car is: {predicted_price[0]}")
