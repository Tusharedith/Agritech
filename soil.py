import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Main Page
st.title("PLANT DISEASE RECOGNITION SYSTEM")

# Introduction
image_path = "C:/Users/tushar swarnkar/Downloads/kknvsd.jpg"
st.image(image_path, use_column_width=True)
st.markdown("""
Welcome to the Plant Disease Recognition System!
""")

# Disease Recognition
st.header("Disease Recognition")
test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
if test_image is not None:
    st.image(test_image, use_column_width=True)
    
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_names = ['Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
                       'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
                       'Tomato__Bacterial_spot', 
                       'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
                       'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
                       'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                       'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_names[result_index]))
