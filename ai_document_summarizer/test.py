import streamlit as st
import pandas as pd
import os
from utils import *


def main():
    st.set_page_config(page_title="PDF Summarizer", page_icon=":guardsman:", layout="wide")
    st.title("PDF Summarizer")
    st.write("Upload a PDF file to get started.")
    st.divider()

    pdf= st.file_uploader("Upload your PDF document", type=["pdf"])
    submit_button = st.button("Generate Summary")
    if submit_button:
        response = summarizer(pdf)
        st.subheader("Summary of the PDF document:")
        st.write(response)
        
        
if __name__ == "__main__":
    main()
