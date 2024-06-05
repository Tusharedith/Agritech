import streamlit as st
import joblib

def main():
    
    st.title("Crop recommendation")


    # Input form for user input
    nitrogen = st.number_input('Nitrogen')
    phosphorus = st.number_input('Phosphorus')
    potassium = st.number_input('Potassium')
    temperature = st.number_input('Temperature')
    humidity = st.number_input('Humidity')
    ph = st.number_input('pH')
    rainfall = st.number_input('Rainfall')

    if st.button('Predict'):
        values = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        
        if 0 < ph <= 14 and temperature < 100 and humidity > 0:
            # Load the trained model
            model = joblib.load('crop_app.pkl')
            arr = [values]
            acc = model.predict(arr)
            st.success(f'The predicted crop yield is: {acc[0]}')
        else:
            st.error("Error in entered values in the form. Please check the values and fill them again.")

if __name__ == '__main__':
    main()