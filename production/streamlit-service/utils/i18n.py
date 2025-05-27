import os
import gettext
import streamlit as st

_supported_langs = ['en', 'ru']


def init_locale(domain: str = "messages", locale_folder: str = "locale"):
    """
    Инициализирует gettext и селектор языка в session_state.
    Возвращает функцию _() для перевода.
    """
    if 'lang' not in st.session_state:
        st.session_state.lang = _supported_langs[0]

    new_lang = st.sidebar.selectbox(
        label="Select language",
        options=_supported_langs,
        index=_supported_langs.index(st.session_state.lang),
        key="lang_selector"
    )
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    base_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir))
    locales_dir = os.path.join(base_dir, locale_folder)

    trans = gettext.translation(
        domain=domain,
        localedir=locales_dir,
        languages=[st.session_state.lang],
        fallback=True
    )
    _ = trans.gettext
    return _
