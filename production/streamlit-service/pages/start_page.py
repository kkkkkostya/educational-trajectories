import streamlit as st
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import json

supported_langs = ['en', 'ru']
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

st.set_page_config(
    layout="centered", page_title="Start page", page_icon="🎓"
)


def main():
    components.html(
        """
        <h1 id="typewriter" style="font-size: 50px; text-align: center; color: #FAFAFA; font-family: sans serif; margin: 0; line-height: 3; margin-bottom: 2em; padding-bottom: 50px;"></h1>
        <script>
        const text = "🎓 Educational trajectories";
        let i = 0;
        function typeWriter() {
            if (i < text.length) {
                document.getElementById("typewriter").innerHTML += text.charAt(i);
                i++;
                setTimeout(typeWriter, 100);
            }
        }
        typeWriter();
        </script>
        """,
        height=150
    )

    try:
        with open("content/start_animation.json", "r", encoding="utf-8") as start_animation:
            animation = json.load(start_animation)
        st_lottie(animation, height=300, key="animation")
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print("Insufficient rights to open the file.")
    except UnicodeDecodeError:
        print("Decoding error: possibly incorrect encoding.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
