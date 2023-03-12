import pickle
import streamlit as st
import numpy as np
import pandas as pd

st.write("""
# Advertising Sales Forecasting App

This app predicts the **Sales forecasting** value!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.0, 500, 250 )
    Radio = st.sidebar.slider('Radio', 0.0, 150, 75)
    Newspaper = st.sidebar.slider('Newspaper', 0.0, 200, 100)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("AdvertisingSVM.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(round(prediction[0],2))
