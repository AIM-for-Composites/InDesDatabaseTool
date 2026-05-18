import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

pages = {
    "": [
        st.Page("page_files/Home.py", title="Home"),
        st.Page("page_files/Categorized_Search.py", title="Categorized Search"),
        st.Page("page_files/Upload_Data.py", title="Upload Data"),
        st.Page("page_files/Contact_Team.py", title="Contact Team"),
    ]
}

pg = st.navigation(pages, position="top")
pg.run()