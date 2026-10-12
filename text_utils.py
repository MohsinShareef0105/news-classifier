"""Shared text preprocessing. Used by the notebook AND the Streamlit app,
so training and inference apply exactly the same cleaning."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STOP = set(ENGLISH_STOP_WORDS)
_URL = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """lowercase -> strip URLs -> keep letters only -> drop stopwords/1-char tokens."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return " ".join(w for w in text.split() if w not in _STOP and len(w) > 1)
