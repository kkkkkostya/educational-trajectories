import streamlit as st
from streamlit_tags import st_tags
from pages.file_reader import read_file

st.set_page_config(
    layout="centered", page_title="Welcome page", page_icon="🎓"
)

def main():
    try:
        content = read_file("content/welcome_content.txt")
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.markdown(content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()