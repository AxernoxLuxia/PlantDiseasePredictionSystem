import streamlit as st
import tensorflow as tf
import numpy as np

def predict(test_image):
    model = tf.keras.models.load_model('PlantDisPredModel.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    inp_arr = tf.keras.preprocessing.image.img_to_array(image)
    inp_arr = np.array([inp_arr])
    pred = model.predict(inp_arr)
    pred_cat = np.argmax(pred)
    return pred_cat

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

#Home Page

if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION")
    st.markdown("This is supposed to be the home page, with some intro")

#About Page

elif(app_mode=="About"):
    st.markdown("This is supposed to be the about page, with some intro")

#Disease Recognition

elif(app_mode=="Disease Recognition"):
    test_image = st.file_uploader("Choose an image: ")
    if(st.button("Show Image")):
        st.image(test_image, use_column_width=True)
    if(st.button("Predict")):
        st.write("Our Prediction")
        with st.spinner("Predicting...."):
            pred_cat = predict(test_image)

            class_name = ['Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy']

        st.success("It is a {}".format(class_name[pred_cat]))