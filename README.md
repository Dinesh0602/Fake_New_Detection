# 📰 Fake News Detection using DeBERTa, XGBoost, and Logistic Regression



## 🔍 Overview

This project aims to detect fake news articles using three distinct machine learning models:
- **DeBERTa** – A transformer-based deep learning model for contextual understanding.
- **XGBoost** – A gradient boosting classifier using sentence-level embeddings.
- **Logistic Regression** – A lightweight baseline model using sentence embeddings.

The final interface is built using **Streamlit** to allow user-friendly real-time predictions.

---

## 👥 Contributors and Responsibilities

| Contributor            | Contribution Areas                                |
|------------------------|---------------------------------------------------|
| **Vinuta Patil**       | Built and fine-tuned the DeBERTa model in `deBERTa/` |
| **Navya Katragada**     | Developed the XGBoost classifier in `XGBoost/`        |
| **Dinesh Buruboyina**  | Implemented Logistic Regression, developed Streamlit UI, and integrated the full system |

All team members collaborated on model evaluation, testing, and documentation.

---

## 🧪 Model Evolution

The project progressed through several modeling phases:

1. **TF-IDF + SVM / Logistic Regression**  
   - Simple but context-agnostic.

2. **LSTM + Word2Vec**  
   - Captured sequential data but lacked generalization and scalability.

3. **Transformers (BERT / DeBERTa)**  
   - Provided state-of-the-art performance with contextual embeddings.



---

## 🔡 Embeddings Used

| Model               | Embedding Type                         |
|---------------------|----------------------------------------|
| **DeBERTa**         | DeBERTa’s own transformer embeddings (contextual) |
| **XGBoost**         | SentenceTransformer embeddings (MPNet) |
| **Logistic Regression** | SentenceTransformer embeddings (MPNet) |

---



## 🖥️ Application Interface (Streamlit UI)

The project includes a **Streamlit-based UI** (`app.py`) where users can input news headlines/text and receive predictions in real-time using the trained models.

### ▶️ To run the app locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## ⚙️ How to Run the Project

1. **Clone the repo**
```bash
git clone https://github.com/Dinesh0602/Fake_New_Detection.git
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure large files are available**
```bash
git lfs install
git lfs pull
```

4. **Run notebooks (optional)**
You can retrain or analyze models via:
- `deBERTa/fake_news_deberta.ipynb`
- `XGBoost/sentenceTransformer_XGBoost.ipynb`
- `Logistic/SentenceTransformer_Logistic.ipynb`

5. **Run the Streamlit UI**
```bash
streamlit run app.py
```

---

## 📊 Model Evaluation

| Model                                                  | Accuracy | F1 Score | Key Take-aways                                                  |
|--------------------------------------------------------|:-------:|:-------:|----------------------------------------------------------------|
| **DeBERTa-v3 Small (Fine-Tuned)**                      | **99.58** | **99.60** | Best overall; deep contextual embeddings drive near-perfect results. |
| **SentenceTransformer + Logistic Regression**          | 91.13   | 91.37   | Strong baseline with fast, lightweight inference.              |
| **SentenceTransformer + XGBoost**                      | 88.43   | 88.44   | Good trade-off between interpretability and performance.       |


---



