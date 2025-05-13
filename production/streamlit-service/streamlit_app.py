import argparse
import streamlit as st

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_file_size", type=float, default=200, help = 'Размер входного файла (в МБ)')
    return parser.parse_args()

def main():
    pg = st.navigation([
        st.Page("pages/start_page.py", title="🎓 Start page"),
        st.Page("pages/welcome_page.py", title="👋 Welcome info"),
        st.Page("pages/prediction_scores_page.py", title="💯 Scores prediction"),
        st.Page("pages/prediction_result.py", title="📉 Prediction result"),
    ])
    pg.run()


if __name__ == "__main__":
    args = parse_args()
    st.session_state["max_file_size"] = args.max_file_size
    main()