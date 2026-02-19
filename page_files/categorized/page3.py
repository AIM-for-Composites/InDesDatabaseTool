import streamlit as st
import pandas as pd
import tabula
import pymupdf
import os
from tqdm import tqdm


def extract_tables_pymupdf(pdf_path):
    """Extract tables using PyMuPDF (alternative method)"""
    try:
        doc = pymupdf.open(pdf_path)
        all_tables = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = page.find_tables()
            
            for table in tables:
                # Extract table data
                table_data = table.extract()
                if table_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    all_tables.append({
                        'page': page_num + 1,
                        'dataframe': df
                    })
        
        doc.close()
        return all_tables
    except Exception as e:
        st.error(f"Error extracting tables with PyMuPDF: {e}")
        return []

def main():
    st.title("PDF Table Extractor")
    st.write("Upload a PDF to extract all tables")
    
    temp_path = "temp_uploaded.pdf"  # Define here
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Using PyMuPDF
        tables = extract_tables_pymupdf(temp_path)
        
        if tables:
            st.success(f"Found {len(tables)} tables!")
            
            for idx, table_info in enumerate(tables):
                st.subheader(f"Table {idx + 1} (Page {table_info['page']})")
                df = table_info['dataframe']
                st.dataframe(df, use_container_width=True)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)