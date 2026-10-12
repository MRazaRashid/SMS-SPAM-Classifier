# import streamlit as st
# import pickle
# import nltk
# import string

# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer


# # -------------------------------
# # Page Configuration
# # -------------------------------
# st.set_page_config(
#     page_title="SMS Spam Detector",
#     page_icon="🛡️",
#     layout="centered"
# )


# # -------------------------------
# # Load Model and Vectorizer
# # -------------------------------
# @st.cache_resource
# def load_artifacts():
#     with open("model.pkl", "rb") as f:
#         model = pickle.load(f)

#     with open("vectorizer.pkl", "rb") as f:
#         vectorizer = pickle.load(f)

#     return model, vectorizer


# model, vectorizer = load_artifacts()


# # -------------------------------
# # NLP Resources
# # -------------------------------
# @st.cache_resource
# def load_nlp_resources():
#     for resource, path in [
#         ("tokenizers/punkt_tab", "punkt_tab"),
#         ("corpora/stopwords", "stopwords")
#     ]:
#         try:
#             nltk.data.find(resource)
#         except LookupError:
#             nltk.download(path, quiet=True)

#     return (
#         set(stopwords.words("english")),
#         PorterStemmer()
#     )


# stop_words, ps = load_nlp_resources()


# # -------------------------------
# # Text Preprocessing
# # -------------------------------
# def trans_text(text):
#     text = text.lower()
#     tokens = nltk.word_tokenize(text)

#     words = []

#     for word in tokens:
#         if word.isalnum() and word not in stop_words:
#             words.append(ps.stem(word))

#     return " ".join(words)


# # -------------------------------
# # Custom Styling
# # -------------------------------
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #f5f7fb;
#     }

#     .main-title {
#         text-align: center;
#         color: #123b32;
#         font-size: 38px;
#         font-weight: 800;
#     }

#     .subtitle {
#         text-align: center;
#         color: #64748b;
#         font-size: 16px;
#         margin-bottom: 25px;
#     }

#     div.stButton > button {
#         background-color: #087f5b;
#         color: white;
#         border-radius: 10px;
#         border: none;
#         height: 48px;
#         font-weight: 600;
#         width: 100%;
#     }

#     div.stButton > button:hover {
#         background-color: #066649;
#         color: white;
#     }
# </style>
# """, unsafe_allow_html=True)


# # -------------------------------
# # Interface
# # -------------------------------
# st.markdown(
#     '<div class="main-title">🛡️ SMS Spam Detector</div>',
#     unsafe_allow_html=True
# )

# st.markdown(
#     '<div class="subtitle">'
#     'An NLP-powered application to detect spam messages'
#     '</div>',
#     unsafe_allow_html=True
# )

# st.divider()

# st.subheader("Analyse your message")

# message = st.text_area(
#     "Enter your SMS message",
#     placeholder="Example: Congratulations! You have won a free prize...",
#     height=150,
#     max_chars=5000
# )

# st.caption(
#     "Your message will be preprocessed, vectorized and classified "
#     "using your trained machine learning model."
# )

# if st.button("🔍 Detect Spam"):

#     if not message.strip():
#         st.warning("Please enter a message first.")

#     else:
#         with st.spinner("Analysing your message..."):

#             # 1. Preprocess text
#             processed_text = trans_text(message)

#             # 2. Convert text into TF-IDF features
#             features = vectorizer.transform([processed_text])

#             # 3. Predict using the trained model
#             prediction = model.predict(features)[0]

#         st.divider()
#         st.subheader("Prediction Result")

#         # LabelEncoder maps ham -> 0 and spam -> 1
#         if prediction == 1:
#             st.error("🚨 SPAM DETECTED")
#             st.write(
#                 "This message has been classified as spam."
#             )

#         else:
#             st.success("✅ NOT SPAM")
#             st.write(
#                 "This message has been classified as a legitimate message."
#             )

#         with st.expander("View preprocessing details"):
#             st.write("**Original message:**")
#             st.write(message)

#             st.write("**Preprocessed message:**")
#             st.code(processed_text or "(No tokens remaining)")

#             st.write("**TF-IDF feature shape:**")
#             st.write(features.shape)
import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered"
)


# -------------------------------
# Load Model and Vectorizer
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_artifacts()


# -------------------------------
# NLP Resources
# -------------------------------
@st.cache_resource
def load_nlp_resources():
    for resource, path in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords")
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(path, quiet=True)

    return (
        set(stopwords.words("english")),
        PorterStemmer()
    )


stop_words, ps = load_nlp_resources()


# -------------------------------
# Text Preprocessing
# -------------------------------
def trans_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    words = []

    for word in tokens:
        if word.isalnum() and word not in stop_words:
            words.append(ps.stem(word))

    return " ".join(words)


# -------------------------------
# Custom Styling (GitHub look)
# Follows the system light / dark setting, like GitHub's "sync with system"
# -------------------------------
st.markdown("""
<style>
    /* ---------- GitHub design tokens: light ---------- */
    :root {
        --gh-bg: #ffffff;
        --gh-subtle: #f6f8fa;
        --gh-input: #ffffff;
        --gh-fg: #1f2328;
        --gh-muted: #636c76;
        --gh-placeholder: #6e7781;
        --gh-border: #d0d7de;
        --gh-border-muted: #d8dee4;
        --gh-accent: #0969da;
        --gh-accent-ring: rgba(9, 105, 218, 0.3);
        --gh-accent-bg: #ddf4ff;
        --gh-header-bg: #f6f8fa;
        --gh-btn: #1f883d;
        --gh-btn-hover: #1a7f37;
        --gh-btn-active: #197935;
        --gh-btn-border: rgba(31, 35, 40, 0.15);
        --gh-btn-shadow: 0 1px 0 rgba(31, 35, 40, 0.1);
        --gh-danger-bg: #ffebe9;
        --gh-danger-border: rgba(255, 129, 130, 0.4);
        --gh-success-bg: #dafbe1;
        --gh-success-border: rgba(74, 194, 107, 0.4);
        --gh-warning-bg: #fff8c5;
        --gh-warning-border: rgba(212, 167, 44, 0.4);
        --gh-font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
                   Helvetica, Arial, sans-serif;
        --gh-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
                   "Liberation Mono", monospace;
    }

    /* ---------- GitHub design tokens: dark ---------- */
    @media (prefers-color-scheme: dark) {
        :root {
            --gh-bg: #0d1117;
            --gh-subtle: #161b22;
            --gh-input: #0d1117;
            --gh-fg: #e6edf3;
            --gh-muted: #7d8590;
            --gh-placeholder: #6e7681;
            --gh-border: #30363d;
            --gh-border-muted: #21262d;
            --gh-accent: #2f81f7;
            --gh-accent-ring: rgba(47, 129, 247, 0.3);
            --gh-accent-bg: rgba(56, 139, 253, 0.1);
            --gh-header-bg: #010409;
            --gh-btn: #238636;
            --gh-btn-hover: #2ea043;
            --gh-btn-active: #238636;
            --gh-btn-border: rgba(240, 246, 252, 0.1);
            --gh-btn-shadow: none;
            --gh-danger-bg: rgba(248, 81, 73, 0.1);
            --gh-danger-border: rgba(248, 81, 73, 0.4);
            --gh-success-bg: rgba(46, 160, 67, 0.15);
            --gh-success-border: rgba(46, 160, 67, 0.4);
            --gh-warning-bg: rgba(187, 128, 9, 0.15);
            --gh-warning-border: rgba(187, 128, 9, 0.4);
        }
    }

    /* ---------- Page ---------- */
    .stApp {
        background-color: var(--gh-bg);
        color: var(--gh-fg);
    }

    .stApp,
    .stApp p,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp summary,
    .stApp textarea,
    .stApp button {
        font-family: var(--gh-font) !important;
    }

    .stApp p,
    .stApp label {
        color: var(--gh-fg);
    }

    .stApp a {
        color: var(--gh-accent);
    }

    .block-container,
    div[data-testid="stMainBlockContainer"] {
        max-width: 760px;
        padding-top: 5rem !important;
    }

    hr {
        border-color: var(--gh-border) !important;
        margin: 24px 0 !important;
    }

    /* ---------- Top navigation bar ---------- */
    div[data-testid="stDecoration"] {
        display: none;
    }

    header[data-testid="stHeader"] {
        background-color: var(--gh-header-bg) !important;
        border-bottom: 1px solid var(--gh-border);
        height: 3.5rem;
    }

    header[data-testid="stHeader"]::before {
        content: "sms-spam-detector";
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        height: 22px;
        line-height: 22px;
        padding-left: 30px;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='%232da44e' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8 1.2 2.2 3.3v3.9c0 3.2 2.3 5.6 5.8 7.1 3.5-1.5 5.8-3.9 5.8-7.1V3.3L8 1.2z'/%3E%3Cpath d='m5.6 8 1.7 1.7 3.2-3.4'/%3E%3C/svg%3E") no-repeat left center / 22px 22px;
        color: var(--gh-fg);
        font-family: var(--gh-font);
        font-size: 14px;
        font-weight: 600;
    }

    /* ---------- Page heading ---------- */
    .main-title {
        text-align: left;
        color: var(--gh-fg);
        font-family: var(--gh-font);
        font-size: 32px;
        font-weight: 600;
        line-height: 1.25;
    }

    .subtitle {
        text-align: left;
        color: var(--gh-muted);
        font-family: var(--gh-font);
        font-size: 16px;
        margin-top: 4px;
        margin-bottom: 12px;
    }

    .topics {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 8px;
    }

    .topic {
        display: inline-block;
        padding: 0 10px;
        border-radius: 2em;
        background-color: var(--gh-accent-bg);
        color: var(--gh-accent);
        font-family: var(--gh-font);
        font-size: 12px;
        font-weight: 500;
        line-height: 22px;
    }

    /* ---------- Section headings (GitHub markdown h2) ---------- */
    div[data-testid="stHeading"] h3 {
        color: var(--gh-fg) !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        line-height: 1.25 !important;
        padding: 0 0 0.3em 0 !important;
        border-bottom: 1px solid var(--gh-border-muted);
    }

    div[data-testid="stHeading"] [data-testid="stHeaderActionElements"] {
        display: none;
    }

    /* ---------- Text area ---------- */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label {
        color: var(--gh-fg) !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"] {
        background-color: var(--gh-input) !important;
        border: 1px solid var(--gh-border) !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="textarea"]:focus-within {
        border-color: var(--gh-accent) !important;
        box-shadow: 0 0 0 3px var(--gh-accent-ring) !important;
        outline: none;
    }

    div[data-baseweb="textarea"] textarea {
        background-color: var(--gh-input) !important;
        color: var(--gh-fg) !important;
        -webkit-text-fill-color: var(--gh-fg);
        font-size: 14px;
        line-height: 20px;
    }

    div[data-baseweb="textarea"] textarea::placeholder {
        color: var(--gh-placeholder) !important;
        -webkit-text-fill-color: var(--gh-placeholder);
        opacity: 1;
    }

    div[data-testid="InputInstructions"] {
        color: var(--gh-muted) !important;
    }

    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p {
        color: var(--gh-muted) !important;
        font-size: 12px !important;
    }

    /* ---------- Primary button (GitHub green) ---------- */
    div.stButton > button {
        background-color: var(--gh-btn) !important;
        color: #ffffff !important;
        border: 1px solid var(--gh-btn-border) !important;
        border-radius: 6px;
        box-shadow: var(--gh-btn-shadow);
        min-height: 40px;
        height: 40px;
        font-size: 14px;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button p {
        color: #ffffff !important;
        font-size: 14px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: var(--gh-btn-hover) !important;
        border-color: var(--gh-btn-border) !important;
        color: #ffffff !important;
    }

    div.stButton > button:active {
        background-color: var(--gh-btn-active) !important;
    }

    div.stButton > button:focus-visible {
        outline: 2px solid var(--gh-accent);
        outline-offset: -2px;
        box-shadow: none;
    }

    /* ---------- Alerts (GitHub flash banners) ---------- */
    div[data-testid="stAlert"] {
        border: 1px solid var(--gh-border);
        border-radius: 6px;
        background-color: var(--gh-subtle);
    }

    div[data-testid="stAlert"] > div,
    div[data-testid="stAlert"] div[data-baseweb="notification"] {
        background-color: transparent !important;
        border: none !important;
        color: var(--gh-fg) !important;
    }

    div[data-testid="stAlert"] p {
        color: var(--gh-fg) !important;
        font-size: 14px;
    }

    div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
        background-color: var(--gh-danger-bg);
        border-color: var(--gh-danger-border);
    }

    div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
        background-color: var(--gh-success-bg);
        border-color: var(--gh-success-border);
    }

    div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
        background-color: var(--gh-warning-bg);
        border-color: var(--gh-warning-border);
    }

    /* ---------- Expander (GitHub box) ---------- */
    div[data-testid="stExpander"] details {
        background-color: var(--gh-subtle) !important;
        border: 1px solid var(--gh-border) !important;
        border-radius: 6px !important;
    }

    div[data-testid="stExpander"] summary {
        color: var(--gh-fg) !important;
        font-size: 14px;
        font-weight: 600;
    }

    div[data-testid="stExpander"] summary:hover {
        color: var(--gh-accent) !important;
    }

    /* ---------- Code block ---------- */
    div[data-testid="stCode"],
    div[data-testid="stCode"] pre {
        background-color: var(--gh-bg) !important;
        border-radius: 6px;
    }

    div[data-testid="stCode"] {
        border: 1px solid var(--gh-border);
    }

    div[data-testid="stCode"] pre,
    div[data-testid="stCode"] code,
    div[data-testid="stCode"] span {
        color: var(--gh-fg) !important;
        font-family: var(--gh-mono) !important;
        font-size: 12px;
    }

    /* ---------- Spinner ---------- */
    div[data-testid="stSpinner"] p {
        color: var(--gh-muted) !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Interface
# -------------------------------
st.markdown(
    '<div class="main-title">🛡️ SMS Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An NLP-powered application to detect spam messages'
    '</div>'
    '<div class="topics">'
    '<span class="topic">nlp</span>'
    '<span class="topic">tf-idf</span>'
    '<span class="topic">spam-detection</span>'
    '<span class="topic">streamlit</span>'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("Analyse your message")

message = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150,
    max_chars=5000
)

st.caption(
    "Your message will be preprocessed, vectorized and classified "
    "using your trained machine learning model."
)

if st.button("🔍 Detect Spam"):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        with st.spinner("Analysing your message..."):

            # 1. Preprocess text
            processed_text = trans_text(message)

            # 2. Convert text into TF-IDF features
            features = vectorizer.transform([processed_text])

            # 3. Predict using the trained model
            prediction = model.predict(features)[0]

        st.divider()
        st.subheader("Prediction Result")

        # LabelEncoder maps ham -> 0 and spam -> 1
        if prediction == 1:
            st.error("🚨 SPAM DETECTED")
            st.write(
                "This message has been classified as spam."
            )

        else:
            st.success("✅ NOT SPAM")
            st.write(
                "This message has been classified as a legitimate message."
            )

        with st.expander("View preprocessing details"):
            st.write("**Original message:**")
            st.write(message)

            st.write("**Preprocessed message:**")
            st.code(processed_text or "(No tokens remaining)")

            st.write("**TF-IDF feature shape:**")
            st.write(features.shape)