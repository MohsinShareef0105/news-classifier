# News Category Classifier

Supervised text-classification pipeline (raw text → preprocessing → TF-IDF → model → evaluation → Streamlit deployment)
trained on the [India Headlines News dataset](https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset).

**Live app:** `<PASTE STREAMLIT URL HERE>`

## Files
| File | Purpose |
|---|---|
| `News_Classification.ipynb` | Full pipeline: EDA, preprocessing, models, evaluation, export |
| `text_utils.py` | Shared cleaning function (training + inference) |
| `app.py` | Streamlit web app |
| `news_classifier.joblib` | Trained pipeline (TF-IDF + classifier) |
| `metrics.json` | Test metrics shown in the app sidebar |
| `requirements.txt` | Pinned deployment dependencies |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
