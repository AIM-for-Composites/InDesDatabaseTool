import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image, ImageOps

st.set_page_config(initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    div[data-baseweb="tab-list"] {
        justify-content: center !important;
    }

    [data-testid="stAppViewContainer"] {
        background-image:
            linear-gradient(120deg, #eaf2ff, #dbeafe, #fdecec, #ffffff),
            linear-gradient(#ffffff, #ffffff);
        background-size:
            300% 560px,
            100% 100%;
        background-position:
            0% 0%,
            0% 0%;
        background-repeat: no-repeat;
        animation: gradientFlow 18s ease infinite;
    }

    @keyframes gradientFlow {
        0%   { background-position: 0% 0%, 0% 0%; }
        50%  { background-position: 100% 0%, 0% 0%; }
        100% { background-position: 0% 0%, 0% 0%; }
    }

    </style>
    """,
    unsafe_allow_html=True
)


tab1, tab2, tab3 = st.tabs(["About", "Explore", "Our Team"])

with tab1:

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        coll1, coll2, coll3, coll4, coll5 = st.columns([1, 1, 2, 1, 1])
        with coll3:
            st.image("logo.png", use_container_width=True)
        st.write("")
    
    st.image("images/Home.png", use_container_width=True)

    st.write("")

    st.markdown(
        """
        <div class="aim-subheading">
        Artificially Intelligent Manufacturing Paradigm (AIM) for Composites
        </div>

        <p>
        The AIM Database tool serves as a powerful, centralized hub designed to streamline collaboration and information
        exchange within the composite materials research community. As illustrated in the diagram, the platform enables
        researchers to actively contribute to a shared knowledge base by directly uploading vital experimental datasets 
        through secure terminals. Users can submit specific measurements regarding mechanical properties, thermal behavior, 
        and rheology, alongside their published journal papers, ensuring that both raw data and peer-reviewed findings are
        integrated into one cohesive system.
        </p>

        <p>
        All contributed information is securely aggregated within a central cloud architecture, allowing for efficient storage,
        organization, and retrieval by authorized users. The database is structured to comprehensively manage data across
        essential material categories, specifically organizing inputs into distinct clusters for polymer data, fiber data,
        and final composite data. By consolidating these diverse resources into a single accessible location, the AIM 
        Database tool empowers scientists to cross-reference findings, avoid duplicating efforts, and ultimately accelerate 
        innovation in the development of advanced materials.    
        </p>
        """,
        unsafe_allow_html=True
    )

with tab2:
    st.write("")

with tab3:
    IMG_DIR = Path(r"C:\Users\varam\Documents\2026_Jan_LatestV_ProjectInDes\Abhi_Code\code\5_2_database\images")
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
        st.markdown("""
        **Gang Li**  
        Professor of Mechanical Engineering, Clemson University  
        gli@clemson.edu
        """)

    with col2:
        st.image(fixed_image("Mathias.jpg"))
        st.markdown("""
        **Heider, Mathias**  
        Research Assistant - CSE PhD Student  
        University of Delaware  
        mheider@udel.edu
        """)

    with col3:
        st.image(fixed_image("Abhijit.jpg"))
        st.markdown("""
        **Abhijit Varanasi**  
        Lab Specialist - Clemson University  
        MFA Graduate, Clemson University  
        BE - CSE  
        avarana@clemson.edu
        """)
    st.write("")

    sp1, col4, sp2, col5, sp3 = st.columns([1, 3, 1, 3, 1])

    with col4:
        st.image(fixed_image("Tejaswi.jpg"))
        st.markdown("""
        **Tejaswi Gudimetla**  
        Lab Aide - Clemson University  
        vgudime@clemson.edu
        """)

    with col5:
        st.image(fixed_image("Pradeep.jpg"))
        st.markdown("""
        **Sai Aditya Pradeep**  
        Research and Development Engineer, University of Delaware  
        spradeep@udel.edu
        """)

st.sidebar.image("logo.png", caption=" ", width=150)
