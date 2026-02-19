import streamlit as st
import pandas as pd
from PIL import Image
import re
import base64

with open("images/Materials_bg_InDeS.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

image_url = f"data:image/png;base64,{encoded}"

st.set_page_config(initial_sidebar_state="expanded")  


def extract_matrix_fiber_from_abbr(abbr: str):
    if not isinstance(abbr, str):
        return None, None

    text = abbr.lower()

    matrix_map = {
        "epoxy": "Epoxy",
        "cyanate ester": "Cyanate Ester",
        "cynate ester": "Cyanate Ester",  
        "polypropylene": "Polypropylene",
        "pp": "Polypropylene",
        "peek": "PEEK",
        "pei": "PEI",
        "nylon": "Nylon",
        "pa6": "PA6",
        "polyester": "Polyester",
        "vinyl ester": "Vinyl Ester",
        "phenolic": "Phenolic"
    }

    matrix = None
    for key, val in matrix_map.items():
        if key in text:
            matrix = val
            break

    fiber_map = {
        "carbon": "Carbon Fiber",
        "glass": "Glass Fiber",
        "e-glass": "E-Glass Fiber",
        "s-glass": "S-Glass Fiber",
        "aramid": "Aramid Fiber",
        "kevlar": "Kevlar Fiber",
        "basalt": "Basalt Fiber",
        "natural": "Natural Fiber"
    }

    fiber = None
    for key, val in fiber_map.items():
        if key in text:
            fiber = val
            break

    return matrix, fiber


def main():
    st.markdown(f"""
        <style>
            .stApp {{
                background-color: #f2f2f2;
            }}

            /* Center the main title */
            h1 {{
                text-align: center;
            }}

            .hero {{
                background-image: linear-gradient(to bottom, rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('{image_url}');
                background-size: cover;
                background-position: center;
                text-align: center;
                padding: 12rem 12rem 12rem 12rem;
                margin: -6rem -6rem 6rem -6rem;
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
            }}
            .hero h3 {{
                font-size: 1.3rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                color: white !important;
            }}
            .hero p {{
                color: white !important;
                font-size: 1rem;
                margin: 0;
            }}
        </style>

        <div class="hero">
            <h3>Categorized Material Search</h3>
            <p>
                Explore the AIM materials database by browsing polymers, fibers, and composites in a structured, category-driven view.<br>
                Use the filters to narrow down materials by class, composition, and property type, then inspect detailed property data and
                associated experimental plots where available.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    mat_section = st.sidebar.expander("Materials", expanded=False)
    with mat_section:
        thermo = mat_section.button("Composites")
        polymers = mat_section.button("Polymers")
        Fibers = mat_section.button("Fibers")

    if "material_type" not in st.session_state:
        st.session_state.material_type = "Composites"

    if thermo:
        st.session_state.material_type = "Composites"
    elif polymers:
        st.session_state.material_type = "Polymers"
    elif Fibers:
        st.session_state.material_type = "Fibers"

    @st.cache_data
    def load_data(material_type):
        file_map = {
            "Composites": "data/Composites_material_data.csv",
            "Polymers": "data/polymers_material_data.csv",
            "Fibers": "data/Fibers_material_data.csv",
        }
        return pd.read_csv(file_map[material_type])
    
    csv_data = load_data(st.session_state.material_type)
    
    CLASS_MAP = {
        "Polymers": "Polymer",
        "Fibers": "Fiber",
        "Composites": "Composite",
    }

    current_class = CLASS_MAP[st.session_state.material_type]

    if "user_uploaded_data" in st.session_state:
        user_df = st.session_state["user_uploaded_data"]
        filtered_user_df = user_df[
            user_df["material_class"] == current_class
        ]
        df = pd.concat([csv_data, filtered_user_df], ignore_index=True)
    else:
        df = csv_data

    st.session_state["base_data"] = df

    st.title("Materials DataSet")
   
    materials_df = (
        df[["material_abbreviation", "material_name"]]
        .fillna("")
        .drop_duplicates()
        .reset_index(drop=True)
    )

    materials_df[["Matrix", "Fiber"]] = materials_df["material_abbreviation"].apply(
        lambda x: pd.Series(extract_matrix_fiber_from_abbr(x))
    )

    def get_selected_value(df, key, column_name):
        if key in st.session_state:
            sel = st.session_state[key]["selection"]["cells"]
            if sel:
                row_idx = sel[0][0]
                return df.iloc[row_idx][column_name]
        return None

    mat = get_selected_value(materials_df, "material_table", "material_abbreviation")

    properties_df = pd.DataFrame(columns=["property_name", "section"])

    prop_col, _ = st.columns([4, 6])
    with st.container(border=True):
        with prop_col:
            with st.expander("Select Property", expanded=bool(mat)):
                if mat:
                    filtered_df = df[
                        (df["material_abbreviation"] == mat) &
                        (df["value"].notna()) &
                        (df["property_name"].notna())
                    ]
                else:
                    filtered_df = df[df["value"].notna() & df["property_name"].notna()]

                property_sel = st.selectbox(
                    "Type of Property",
                    filtered_df["section"].drop_duplicates()
                )

                properties_df = (
                    filtered_df[filtered_df["section"] == property_sel][["property_name", "section"]]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

                st.dataframe(
                    properties_df.style.set_properties(**{
                        'background-color': 'white',
                        'color': "#c8c0c0",
                        'border': '1px solid #e0e0e0',
                        'font-family': 'monospace',
                        'font-size': '12px'
                    }).set_table_styles([
                        {
                            'selector': 'thead tr th',
                            'props': [
                                ('background-color', '#f5f5f5'),
                                ('color', "#C8C4C4"),
                                ('font-weight', '600'),
                                ('border-bottom', '2px solid #d0d0d0'),
                                ('font-family', 'monospace'),
                            ]
                        },
                        {
                            'selector': 'tbody tr:hover',
                            'props': [('background-color', '#f9f9f9')]
                        }
                    ]),
                    key="property_table",
                    selection_mode="single-cell",
                    on_select="rerun",
                    use_container_width=True,
                    height=300
                )

    selected_matrix = "All"
    selected_fiber = "All"

    if st.session_state.material_type == "Composites":
        matrix_options = sorted(materials_df["Matrix"].dropna().unique())
        fiber_options = sorted(materials_df["Fiber"].dropna().unique())

        fcol1, fcol2, _ = st.columns([2, 2, 6])
        with fcol1:
            selected_matrix = st.selectbox("Matrix Material", ["All"] + matrix_options)
        with fcol2:
            selected_fiber = st.selectbox("Fiber Material", ["All"] + fiber_options)

    filtered_materials_df = materials_df.copy()

    if st.session_state.material_type == "Composites":
        if selected_matrix != "All":
            filtered_materials_df = filtered_materials_df[
                filtered_materials_df["Matrix"] == selected_matrix
            ]

    if selected_fiber != "All":
        filtered_materials_df = filtered_materials_df[
            filtered_materials_df["Fiber"] == selected_fiber
        ]

    st.dataframe(
        filtered_materials_df.style.set_properties(**{
            'background-color': 'white',
            'color': "#d3cccc",
            'border': '1px solid #e0e0e0',
            'font-family': 'monospace',
            'font-size': '12px'
        }).set_table_styles([
            {
                'selector': 'thead tr th',
                'props': [
                    ('background-color', '#f5f5f5'),
                    ('color', "#D3C9C9"),
                    ('font-weight', '600'),
                    ('border-bottom', '2px solid #d0d0d0'),
                    ('font-family', 'monospace'),
                ]
            },
            {
                'selector': 'tbody tr:hover',
                'props': [('background-color', '#f9f9f9')]
            }
        ]),
        key="material_table",
        selection_mode="single-cell",
        on_select="rerun",
        use_container_width=True,
        height=500
    )

    prop = get_selected_value(properties_df, "property_table", "property_name")

    st.write("")
    if st.button("Search", disabled=not (mat and prop)):
        st.write(f"**Material:** {mat}")
        st.write(f"**Property:** {prop}")

        result = df[
            (df["material_abbreviation"] == mat) &
            (df["property_name"] == prop) &
            (df["value"].notna())
        ]

        if not result.empty:
            st.subheader("Property Data")
            st.dataframe(result.T, use_container_width=True)

            st.subheader("Property Graph")
            img_path = f"images/{mat}_{prop}.png"

            try:
                img = Image.open(img_path)
                st.image(img, use_container_width=True, caption="Stress strain curve")
            except FileNotFoundError:
                st.write("")
        else:
            st.warning("No data found for this material-property combination")