import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import plotly.express as px

#title
st.title('Classification model for transports')

#upload the file
file = st.file_uploader('Upload photo', type=['png', 'jpeg', 'gif', 'jpg','jfif'])
if file:
    st.image(file)

    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('transpot_model.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Extimollik: {probs[pred_id]*100:.1f}%')

    #plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)