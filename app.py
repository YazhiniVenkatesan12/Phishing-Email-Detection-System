import streamlit as st
import joblib
from scipy.sparse import hstack
from PIL import Image
import numpy as np

# -- PAGE CONFIGURATION --
st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- SIDEBAR: About/Instructions --
with st.sidebar:
    st.title("About This App")
    st.write(
        """
        **Phishing Email Detection System**  
        This tool uses an advanced ensemble of machine learning models to detect phishing emails with high accuracy.

        - Paste any email content and click "Detect".
        - See a detailed result, including confidence scores.

        **How it works:**  
        The system combines Random Forest, XGBoost, and Logistic Regression stacked on specialized text features.
        """
    )
    
    st.markdown("---")
    st.info("Developed by Yazhini V | Powered by Streamlit")

# -- HEADER/LOGO --
st.markdown(
    """
    <div style='display:flex; align-items:center; justify-content:center; gap:20px; margin-bottom: 20px;'>
        <img src='https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f6e1.png' width='48'/>
        <div>
            <span style='font-size:2.5em; font-weight:bold;'>Phishing Email Detection System</span><br>
            <span style='font-size:1.1em; color: #ccc;'>An Intelligent, Ensemble-Based Email Classifier</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -- LOAD MODELS/VECTORIZERS --
@st.cache_resource
def load_artifacts():
    word_vectorizer = joblib.load('word_vectorizer.pkl')
    char_vectorizer = joblib.load('char_vectorizer.pkl')
    stacked_model = joblib.load('stacked_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return word_vectorizer, char_vectorizer, stacked_model, label_encoder

word_vectorizer, char_vectorizer, stacked_model, label_encoder = load_artifacts()

# -- INPUT AREA (WITH PLACEHOLDER) --
st.markdown("### Paste Email Content")
email_text = st.text_area(
    "",
    height=180,
    placeholder="Paste the body of the email you want to check here...",
)

detect_btn = st.button("🔍 Detect", type="primary")

# -- PROCESS AND DISPLAY RESULTS --
if detect_btn and email_text.strip():
    # Vectorize input
    X_word = word_vectorizer.transform([email_text])
    X_char = char_vectorizer.transform([email_text])
    X_tfidf = hstack([X_word, X_char])
    # Predict
    probs = stacked_model.predict_proba(X_tfidf)[0]
    pred_label_index = np.argmax(probs)
    pred_label = label_encoder.classes_[pred_label_index]

    # -- RESULT CARD --
    st.markdown("#### Detection Result")
    result_col, conf_col = st.columns([2,1])
    with result_col:
        if pred_label == "Phishing Email":
            st.error(
                "**Phishing Email Detected!**",
                icon="🚨"
            )
            st.markdown(
                "<span style='color:#e74c3c;font-size:1.2em;'><strong>Do NOT click on any links or provide personal information.</strong></span>",
                unsafe_allow_html=True,
            )
        else:
            st.success(
                "**No Phishing Detected**",
                icon="✅"
            )
            st.markdown(
                "<span style='color:#27ae60;font-size:1.2em;'>This email appears safe, but always double-check!</span>",
                unsafe_allow_html=True,
            )
    with conf_col:
        st.markdown("##### Confidence")
        for idx, class_name in enumerate(label_encoder.classes_):
            bar_color = "#e74c3c" if class_name == "Phishing Email" else "#27ae60"
            st.markdown(
                f"<div style='margin-bottom:8px;'>"
                f"<strong>{class_name}:</strong> "
                f"<span style='color:{bar_color};font-weight:bold;'>{probs[idx]*100:.2f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(probs[idx]), text=class_name)

    # -- SUGGESTIONS/EXTRA INFO --
    st.markdown("---")
    st.info(
        "If you believe this is a false positive/negative, please review the email carefully and report as needed."
    )

elif detect_btn:
    st.warning("Please paste or enter email content before clicking Detect.")

# -- FOOTER --
st.markdown(
    """
    <hr style="margin-top:2em;margin-bottom:.5em;">
    <div style='text-align:center'>
        <small>
            &copy; 2025 Yazhini V &mdash; Phishing Email Detection System.<br>
            Built with <a href='https://streamlit.io' target='_blank'>Streamlit</a>.
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)
