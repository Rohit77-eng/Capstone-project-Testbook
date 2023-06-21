import streamlit as st
import pandas as pd 
import numpy as np 
import pickle

model = pickle.load(open('dt_classifier.pkl', 'rb'))
# Set the title of the app
st.title('Car Selling Price Prediction')

