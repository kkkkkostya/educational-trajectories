import streamlit as st
from streamlit_tags import st_tags
from pages.file_reader import read_file
import pandas as pd
import io

st.set_page_config(
    layout="centered", page_title="Preduction result", page_icon="📉"
)

def main():
    st.markdown("<h1 style='font-size: 45px;'>📉 Results</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 30px;'>Here you can see the result of the model's work and download the processed report 🎈</h1>", unsafe_allow_html=True)

    if "result_data" not in st.session_state:
        st.warning("First, upload the file and click Submit on the Scores prediction page...")
        st.stop()
    
    df_pred = st.session_state["result_data"]

    st.dataframe(df_pred)

    csv_buffer = df_pred.to_csv(index=False, sep=",").encode("utf-8")
    txt_buffer = df_pred.to_csv(index=False, sep="\t").encode("utf-8")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_pred.to_excel(writer, index=False, sheet_name="Sheet1")
    excel_data = excel_buffer.getvalue()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📥 Download as CSV",
            data=csv_buffer,
            file_name="data.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            label="📥 Download TXT (TSV)",
            data=txt_buffer,
            file_name="data.txt",
            mime="text/tab-separated-values"
        )
    
    with col3:
        st.download_button(
            label="📥Download Excel",
            data=excel_data,
            file_name="data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()