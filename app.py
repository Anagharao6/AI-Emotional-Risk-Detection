import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Emotional Risk Detection",
    page_icon="🧠",
    layout="centered"
)

# ---------------- TITLE ----------------

st.title("🧠 AI-Based Emotional Risk Detection")

st.write(
    "Enter a message below to classify its predefined emotional-risk category."
)

st.divider()

# ---------------- LOAD DATASET ----------------

data = pd.read_csv("data/dataset.csv")

X = data["text"]
y = data["risk"]

# ---------------- TEXT PREPROCESSING ----------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

X = X.apply(clean_text)

# ---------------- MODEL ----------------

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression())
])

model.fit(X, y)

# ---------------- EXAMPLE MESSAGES ----------------

st.subheader("💡 Try an example")

example = st.selectbox(
    "Choose a sample message:",
    [
        "Select an example",
        "I am having a great day and feeling happy",
        "I am worried about my exams and feeling stressed",
        "I feel completely hopeless and helpless"
    ]
)

# ---------------- USER INPUT ----------------

message = st.text_area(
    "Enter your message:",
    value="" if example == "Select an example" else example,
    placeholder="Type a message here..."
)

# ---------------- ANALYZE ----------------

if st.button("🔍 Analyze Message", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        cleaned_message = clean_text(message)

        prediction = model.predict([cleaned_message])[0]

        probabilities = model.predict_proba([cleaned_message])[0]
        confidence = max(probabilities) * 100

        st.divider()
        st.subheader("📊 Result")

        if prediction == "low":
            st.success("🟢 Risk Category: LOW")

        elif prediction == "medium":
            st.warning("🟡 Risk Category: MEDIUM")

        else:
            st.error("🔴 Risk Category: HIGH")

        st.write(f"**Model confidence:** {confidence:.2f}%")

        st.progress(confidence / 100)

        st.info(
            "This is an educational machine-learning prototype. "
            "It does not provide a medical or psychological diagnosis."
        )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Built using Python, TF-IDF, Logistic Regression and Streamlit."
)