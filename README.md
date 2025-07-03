# User-Frustration-Detector

This project uses **BERT-based deep learning** to analyze tech product reviews—specifically wireless earphones—and **predict if a user is frustrated**. It also includes a **Streamlit web app** for real-time predictions and visual analysis.

## 🔍 Problem Statement

Many users leave product reviews expressing dissatisfaction or frustration, but it’s hard to analyze them at scale. This project helps detect **frustrated users automatically** based on their review text.

## ✅ Features

- Fine-tuned **BERT (bert-base-uncased)** model for binary classification.
- Cleaned and labeled dataset of wireless earphone reviews.
- **Streamlit app** for:
  - Single or bulk review prediction
  - Highlighting trigger words
  - Visualization of frustration trends
- Support for uploading `.csv` files with reviews.

## 🧠 Tech Stack

- **Natural Language Processing** (NLP)
- **Hugging Face Transformers** (BERT)
- **PyTorch**, **Datasets**
- **Streamlit** (for web app)
- **Pandas**, **Matplotlib**, **scikit-learn**

## 📁 Project Structure
```text
User Frustration Detector/
├── User_frustration_app/
│   ├── app.py                # Streamlit app
│   └── saved_model/          # Model used by the app
├── data/                      # Review dataset (CSV)
└── README.md
├── User_Frustration_Project(2).ipynb  # Jupyter notebook
├── model.h5                  # Saved Keras-compatible model
├── requirements.txt

```
## 🚀 How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```
2. Run Jupyter Notebook
The notebook covers:
 - Data loading
 - Tokenization
 - Model training
 - Evaluation

3. Run the Streamlit App
```cd User_frustration_app
streamlit run app.py
```
