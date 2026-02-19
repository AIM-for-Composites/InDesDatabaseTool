import streamlit as st

st.set_page_config(initial_sidebar_state="expanded")  

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
        display: block !important;
    }

    
    /* Hide the collapse button */
    button[kind="header"][data-testid="baseButton-header"] {
        display: none !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stVerticalBlock"] {
        border-color: black !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




def load_page1():
    from page_files.categorized.page1 import main
    main()
    
# def load_page2():
#     from pages.categorized.page2 import main
#     main()



load_page1()

    
#st.sidebar.button('Material Type', on_click=load_page1)
#st.sidebar.button('Trade Name', on_click=load_page2)
#st.sidebar.button('Manufacturer Name', on_click=load_page3)

#image = Image.open('logo.png')
#st.image(image, caption='a', use_container_width=True)
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.image("logo.png", caption=" ", width=150)

