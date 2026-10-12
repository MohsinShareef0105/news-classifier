import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import text_utils  # noqa: F401  (needed so the pickled pipeline can resolve clean_text)

BASE = Path(__file__).parent
st.set_page_config(page_title="News Category Classifier", page_icon="📰", layout="centered")

CATEGORY_INFO = {
    "city": "City / local civic news",
    "india": "National (India) news",
    "world": "International news",
    "sports": "Sports",
    "entertainment": "Entertainment & cinema",
    "business": "Business & economy",
    "education": "Education",
    "life-style": "Lifestyle, health & fitness",
    "tech": "Technology",
    "tv": "Television",
}

EXAMPLES = {
    "Cricket": "Kohli hits century as India beat Australia in the second Test at Melbourne",
    "Business": "Sensex jumps 500 points as RBI keeps repo rate unchanged, banking stocks rally",
    "Entertainment": "Shah Rukh Khan's new film breaks opening day box office records",
    "World": "UN Security Council holds emergency meeting over escalating conflict in the Middle East",
    "Education": "CBSE announces Class 12 board exam results, pass percentage rises this year",
    "Tech": "Apple unveils new iPhone with faster chip and improved camera at launch event",
}


@st.cache_resource
def load_model():
    return joblib.load(BASE / "news_classifier.joblib")


@st.cache_data
def load_metrics():
    p = BASE / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


model = load_model()
metrics = load_metrics()

st.title("📰 News Category Classifier")
st.caption("Paste a news headline or article. The model predicts its news category "
           "(TF-IDF + linear classifier trained on the India Headlines News dataset).")

if "text" not in st.session_state:
    st.session_state.text = ""

st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, (name, txt) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.text = txt

text = st.text_area("News text", key="text", height=160,
                    placeholder="Enter a news headline or article here...")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        probs = model.predict_proba([text])[0]
        classes = model.classes_
        order = probs.argsort()[::-1]
        top = classes[order[0]]
        st.success(f"Predicted category: **{top}**  ({probs[order[0]]:.1%} confidence)")
        st.caption(CATEGORY_INFO.get(top, ""))
        df = pd.DataFrame({"Category": classes[order][:5], "Probability": probs[order][:5]})
        st.bar_chart(df.set_index("Category"))
        with st.expander("Show all class probabilities"):
            st.dataframe(df.assign(Probability=df.Probability.map("{:.2%}".format)), hide_index=True)

if metrics:
    with st.sidebar:
        st.header("Model info")
        st.write(f"**Model:** {metrics['best_model']}")
        st.write(f"**Test accuracy:** {metrics['accuracy']:.3f}")
        st.write(f"**Macro F1:** {metrics['macro_f1']:.3f}")
        st.write(f"**Classes ({len(metrics['classes'])}):** " + ", ".join(metrics["classes"]))
        st.caption("Note: trained on short headlines, so shorter inputs work best.")
