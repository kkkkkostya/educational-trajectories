import streamlit as st
from pages.file_reader import read_file
import pandas as pd
import numpy as np
import requests

MAX_SIZE_FILE = st.session_state.get("max_file_size", 200)
ip_api = "0.0.0.0"
port_api = "5000"

selected_language = st.session_state["language"]

st.set_page_config(
    layout="centered", page_title="Scores prediction", page_icon="🎓"
)


def main():
    st.markdown("<h1 style='font-size: 45px; text-align: center;'>💯 Scores prediction</h1>",
                unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 30px;'></div>",
                unsafe_allow_html=True)
    st.subheader("Upload your Grade report file here")
    uploaded_file = st.file_uploader(
        label="Choose a file",
        type=['csv', 'xlsx', 'xls', 'txt'],
        help=f"Allowed formats: csv, excel (xls, xlsx), txt. Max size: {MAX_SIZE_FILE} MB"
    )

    try:
        table_struct_expl = read_file("content/table_structure_expl.txt")
    except Exception as e:
        st.error(str(e))
        st.stop()
    with st.expander("Explanation of supported file formats and the structure required in them", expanded=False):
        st.markdown(table_struct_expl)

    if uploaded_file is not None:
        file_size = uploaded_file.size
        max_size = MAX_SIZE_FILE * 1024 * 1024
        if file_size > max_size:
            st.error(
                f"File size exceeds {MAX_SIZE_FILE} MB limit. Your file is {file_size / (1024*1024):.2f} MB.")
            return

    st.subheader("Select model")
    options = st.selectbox(
        label="Choose prediction model:",
        options=[
            "CARTE",
        ]
    )

    st.subheader("Indicate the possible range of grades")

    col1, col2 = st.columns(2)
    with col1:
        start_val = st.number_input(
            label="Min grade",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
            format="%d"
        )
    with col2:
        end_val = st.number_input(
            label="Max grade",
            min_value=0,
            max_value=1000,
            value=100,
            step=1,
            format="%d"
        )

    if start_val >= end_val:
        # отображаем ошибку
        st.error("❗️Error:  Min grade should be less than Max grade.")
        st.stop()

    table_separaton = st.text_input(
        "Еnter table separator", value=",", max_chars=10)
    table_heading = st.checkbox("Is there a heading in the table?", value=True)
    table_index = st.checkbox(
        "Is there a index column in the table?", value=True)
    integer_grades_flag = st.checkbox(
        "There can only be integer estimates?", value=True)

    if st.button("✅ Submit"):
        if uploaded_file is None:
            st.warning("Please upload a file before submitting.")
            return
        if not options:
            st.warning("Please select at least one option.")
            return

        params = {'header': None if not table_heading else 0,
                  'index_col': False if not table_index else 0}

        file_extension = uploaded_file.name.split('.')[-1].lower()
        try:
            if file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(uploaded_file, **params)
            elif file_extension == 'csv':
                df = pd.read_csv(uploaded_file, sep=table_separaton, **params)
            elif file_extension == 'txt':
                df = pd.read_csv(uploaded_file, sep=table_separaton, **params)
            else:
                st.error("Unsupported file format.")
                return
            if not table_heading:
                df.columns = ["lesson_"+str(i)
                              for i in range(1, df.shape[1]+1)]
            df = df.replace({np.nan: None})
        except Exception as e:
            st.error(f"Error during reading file: {e}")
            return

        try:
            request_data = {"data": df.to_dict(
                orient="records"), "min_value": start_val, "max_value": end_val, "integer_grades_flag": integer_grades_flag}
            response = requests.post(
                f"http://{ip_api}:{port_api}/predict_carte_model", json=request_data)

            if response.status_code == 200:
                new_df = pd.DataFrame(response.json())
                new_df = new_df.replace({None: np.nan})

                print(new_df)

                st.session_state["result_data"] = new_df
                st.switch_page("pages/prediction_result.py")
            else:
                st.error(
                    f"Request failed with status code {response.status_code}")
        except Exception as e:
            st.error(f"Error processing data with ML model: {e}")
            return


if __name__ == "__main__":
    main()
