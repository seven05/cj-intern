import streamlit as st
st.header("st.selectbox")

CONFIDENCE = st.slider(
    "Confidence score",
    0, 100, 25
)

option = st.selectbox(
    'Model weights',
    ("face.pt", "player.pt")
)


video_file = open('../test_ufc.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)