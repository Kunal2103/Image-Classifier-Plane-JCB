
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import joblib
from PIL import Image

st.set_page_config(page_title='Image Clasifier',page_icon="📸")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


st.markdown(f'<b><h1 style="color:#707979;font-size:15px;">{"Created by: Krunal Pandya"}</h1></b>', unsafe_allow_html=True)



html_temp = """
<div style="background-color:#707979;padding:2px">
<h2 style="color:white;text-align:center;">Image Classifier: Aeroplane / JCB</h2>
</div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#9CA3A6;font-size:24px;">{"Upload an Image"}</h1>', unsafe_allow_html=True)

model = joblib.load(open('aer-jcb.p','rb'))


uploaded_file = st.file_uploader("Choose a JPG / PNG format.......",type=['jpg','png'])

primaryColor = st.get_option("theme.primaryColor")

s = f"""
<style>
div.stButton > button:first-child {{ border: 2px solid {primaryColor}; border-radius:10px 10px 10px 10px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img,caption="Uploaded Image")
    st.markdown("""---""")

    
    if st.button("PREDICT"):
        CATEGORIES = ['Aeroplane','JCB']
        #st.write("Result...")
        flat_data =[]
        img = np.array(img)
        img_resized = resize(img,(150,150,3))
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)
        y_out = model.predict(flat_data)
        y_out = CATEGORIES[y_out[0]]
        st.title(f'Predicted Image: {y_out}')

        
