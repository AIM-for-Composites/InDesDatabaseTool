import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps

IMG_DIR = Path("images")
TARGET_SIZE = (213, 310)

def fixed_image(name):
    img = Image.open(IMG_DIR / name).convert("RGB")
    return ImageOps.fit(img, TARGET_SIZE, Image.LANCZOS, centering=(0.5, 0.5))

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.subheader("Team Members")

col1, col2, col3 = st.columns(3)
with col1:
    st.image(fixed_image("GangLi.jpg"))
    st.markdown("**Gang Li**  \nProfessor of Mechanical Engineering, Clemson University  \ngli@clemson.edu")

with col2:
    st.image(fixed_image("Mathias.jpg"))
    st.markdown("**Heider, Mathias**  \nResearch Assistant - CSE PhD Student  \nUniversity of Delaware  \nmheider@udel.edu")

with col3:
    st.image(fixed_image("Abhijit.jpg"))
    st.markdown("**Abhijit Varanasi**  \nLab Specialist - Clemson University  \nMFA Graduate, Clemson University  \nBE - CSE  \navarana@clemson.edu")

st.write("")

sp1, col4, sp2, col5, sp3 = st.columns([1, 3, 1, 3, 1])
with col4:
    st.image(fixed_image("Tejaswi.jpg"))
    st.markdown("**Tejaswi Gudimetla**  \nLab Aide - Clemson University  \nvgudime@clemson.edu")

with col5:
    st.image(fixed_image("Pradeep.jpg"))
    st.markdown("**Sai Aditya Pradeep**  \nResearch and Development Engineer, University of Delaware  \nspradeep@udel.edu")

st.sidebar.image("logo.png", caption=" ", width=150)
