import streamlit as st
import numpy as np
import pickle
# import sklearn

# Load models
model = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Title and description
st.title('Crop Yield Prediction')
st.write('Enter the input values to predict the crop yield:')

# Input form for user input
Year = st.number_input('Year')
average_rain_fall_mm_per_year= st.number_input('Rainfall')
pesticides_tonnes= st.number_input('Pesticides')
avg_temp= st.number_input('Temperature')
Area= st.text_input('Area')
Item = st.text_input('Item')



# Button to predict
if st.button('Predict'):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
    transformed_features = preprocessor.transform(features)
    prediction = model.predict(transformed_features).reshape(1, -1)
    st.success(f'The predicted crop yield is: {prediction[1]}')