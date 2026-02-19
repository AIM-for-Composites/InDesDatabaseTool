import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state = "collapsed")




st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f0f0;
    }
    
    

    body, p, div, span, h1, h2, h3, .stMarkdown {
        color: black !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: black !important;
    }
    
    [data-testid="stVerticalBlock"] {
        border-color: black !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: black !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    header a, header button, header span, header div, header p {
        color: white !important;
    }
    
    /* CENTER NAVIGATION - Target all possible containers */
    [data-testid="stPageNav"],
    [data-testid="stPageNav"] > div,
    [data-testid="stPageNav"] > div > div,
    [data-testid="stPageNav"] ul,
    [data-testid="stPageNav"] nav {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    /* Remove any left margin/padding that might offset it */
    [data-testid="stPageNav"] * {
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Style the links */
    [data-testid="stPageNav"] a {
        color: white !important;
        background-color: black !important;
        border: 2px solid white !important;
        padding: 10px 30px !important;
        border-radius: 5px !important;
        margin: 0 10px !important;
        font-weight: 600 !important;
        text-decoration: none !important;
    }
    
    /* Active page styling */
    nav a[aria-current="page"] {
        background-color: white !important;
        color: black !important;
    }
    
    [data-testid="stPageNav"], [data-testid="stPageNav"] * {
        color: white !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    
    /* Make sidebar toggle button visible */
    button[kind="header"] {
        color: white !important;
        background-color: black !important;
    }
    /* DROPDOWN STYLING - Light gray background, black text */
    div[data-baseweb="select"] {
        background-color: #e8e8e8 !important;
    }
    
    div[data-baseweb="select"] * {
        color: black !important;
        background-color: #e8e8e8 !important;
    }
    
    /* Dropdown menu options */
    ul[role="listbox"] {
        background-color: #f5f5f5 !important;
    }
    
    li[role="option"] {
        background-color: #f5f5f5 !important;
        color: black !important;
    }
    
    li[role="option"]:hover {
        background-color: #d0d0d0 !important;
        color: black !important;
    }
    
    
    
    </style>
    """,
    unsafe_allow_html=True
)



pages = {
    "": [  
        st.Page("page_files/Home.py", title="Home"),
        st.Page("page_files/Categorized_Search.py", title="Categorized Search"),
        st.Page("page_files/Upload_Data.py", title="Upload Data"),
    ]
}

pg = st.navigation(pages, position="top")
pg.run()