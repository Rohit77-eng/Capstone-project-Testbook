import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained machine learning model
model = pickle.load(open('dt_classifier.pkl', 'rb'))

# Load the car details dataset
car_details = pd.read_csv('CAR DETAILS.csv')

# Get the minimum and maximum values for each feature
def scale_input(value, feature):
    min_val = car_details[feature].min()
    max_val = car_details[feature].max()
    scaled_value = (value - min_val) / (max_val - min_val)
    return scaled_value

# Function to preprocess input features and predict the selling price
def predict_price(features):
    scaled_features = features.copy()
    scaled_features[0] = scale_input(features[0], 'year')

    predicted_price = model.predict([scaled_features])

    return predicted_price

# Streamlit app
def main():
    st.title("Car Selling Price Prediction")
    st.write("Enter the following details to get a predicted selling price.")

    # Input features
    year = st.sidebar.slider('Year', min_value=int(car_details['year'].min()), max_value=int(car_details['year'].max()), step=1)
    kilometers_driven = st.sidebar.number_input('Kilometers Driven', min_value=int(car_details['km_driven'].min()), max_value=int(car_details['km_driven'].max()), step=500)
    fuel = st.selectbox("Fuel", car_details['fuel'].unique())
    seller_type = st.selectbox("Seller Type", car_details['seller_type'].unique())
    transmission = st.selectbox("Transmission", car_details['transmission'].unique())
    owner = st.selectbox("Owner", car_details['owner'].unique())

    if st.button("Predict"):
        # Prepare features
        features = [year, kilometers_driven, fuel, seller_type, transmission, owner]
        fuel_dict = {"Petrol": 0, "Diesel": 1, "CNG": 2, "LPG": 3, "Electric": 4}
        seller_dict = {"Individual": 0, "Dealer": 1, "Trustmark Dealer": 2}
        transmission_dict = {"Manual": 0, "Automatic": 1}
        owner_dict = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2, "Fourth & Above Owner": 3, "Test Drive Car": 4}
        features[2] = fuel_dict[features[2]]
        features[3] = seller_dict[features[3]]
        features[4] = transmission_dict[features[4]]
        features[5] = owner_dict[features[5]]

        # Call the predict_price function to get the predicted price
        predicted_price = predict_price(features)

        # Display the predicted price
        st.success(f"The predicted selling price is {predicted_price[0]}")

if __name__ == "__main__":
    main()