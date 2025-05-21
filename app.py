import streamlit as st
st.set_page_config(page_title="Fake-News Detector (offline)",
                   layout="centered") 
import numpy as np
import joblib, torch, xgboost as xgb
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from sentence_transformers import SentenceTransformer

# ---------- 1. One-time model loading ---------------------------------
@st.cache_resource(show_spinner="Loading models…")
def load_all():
    # A) MPNet embedder + logistic regression
    mpnet = SentenceTransformer("all-mpnet-base-v2")
    logreg = joblib.load("Mpnet/mpnet_classifier.pkl")

    # B) MiniLM embedder + XGBoost (vectoriser is another SBERT model)
    minilm = SentenceTransformer("all-MiniLM-L6-v2")
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model("XGBoost/xgb_model.json")

    # C) DeBERTa-v3-small fine-tuned
    tok = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")
    deberta = DebertaV2ForSequenceClassification.from_pretrained(
        "deBERTa"
    )

    return (mpnet, logreg), (minilm, xgb_clf), (tok, deberta)


(mpnet_model, mpnet_clf), (minilm_model, xgb_clf), (deberta_tok, deberta_model) = load_all()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deberta_model.to(device).eval()

# ---------- 2. Helper inference functions -----------------------------
def predict_mpnet(text: str):
    emb = mpnet_model.encode([text], show_progress_bar=False)
    proba = mpnet_clf.predict_proba(emb)[0]
    return int(proba.argmax()), float(proba.max())

def predict_xgb(text: str):
    feats = minilm_model.encode([text], show_progress_bar=False)
    proba = xgb_clf.predict_proba(feats)[0]
    return int(proba.argmax()), float(proba.max())

@torch.no_grad()
def predict_deberta(text: str):
    inputs = deberta_tok(text,
                         return_tensors="pt",
                         truncation=True,
                         padding=True,
                         max_length=512).to(device)
    logits = deberta_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(probs.argmax()), float(probs.max())

# Map for the UI control
PREDICTORS = {
    "MPNet + Logistic Reg": predict_mpnet,
    "XGBoost (MiniLM feats)": predict_xgb,
    "DeBERTa-v3-small": predict_deberta,
}

# ---------- 3. Streamlit UI -------------------------------------------

st.title("📰 Fake-News Detector – all local, no server needed")

model_name = st.selectbox("Choose model:", list(PREDICTORS.keys()))
text = st.text_area("Paste the news article here:", height=250)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner(f"Running {model_name}…"):
        label, conf = PREDICTORS[model_name](text)

    st.success("Done!")
    st.markdown(f"**Result:** {'❌ Fake' if label else '✅ Real'}")
    st.markdown(f"**Confidence:** {conf*100:.1f}%")
