import streamlit as st


st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f0f0;
    }
    body, p, div, span, h1, h2, h3 {
        color: black !important;
    }
    
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    header[data-testid="stHeader"] {
        background-color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.markdown(
#     """
#     <style>
#     section[data-testid="stSidebar"] {
#         display: none;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# def load_page1():
#     from pages.categorized.page1 import main
#     main()

def load_page6():
    from page_files.categorized.page6 import main
    main()

def load_page3():
    from page_files.categorized.page3 import main
    main()

load_page6()
#load_page3()
