\# AI-Based Emotional Risk Detection



An educational machine-learning project that classifies user-provided text into predefined emotional-risk categories: Low, Medium, or High.



\## Features



\- Text preprocessing using Python

\- TF-IDF feature extraction

\- Logistic Regression classification

\- Three predefined risk categories

\- Interactive Streamlit web interface

\- Model confidence display



\## Technologies Used



\- Python

\- Pandas

\- Scikit-learn

\- TF-IDF

\- Logistic Regression

\- Streamlit



\## How It Works



1\. User enters a text message.

2\. The text is cleaned and preprocessed.

3\. TF-IDF converts the text into numerical features.

4\. Logistic Regression predicts the predefined risk category.

5\. The application displays the predicted category and model confidence.



\## Project Structure



```text

AI-Emotional-Risk-Detection/

│

├── data/

│   └── dataset.csv

├── app.py

├── create\_dataset.py

├── train\_model.py

├── requirements.txt

└── .gitignore

