import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import pandas as pd
import plotly.express as px
import re

# ======================= CONFIG ============================
MAX_LEN = 512

# Frustration keywords
frustration_keywords = [
    "waste", "useless", "not working", "not functioning", "doesn't work", "isn't working",
    "broken", "damaged", "stopped", "crashed", "crash", "hang", "lag", "freezing", "froze", "slow",
    "disappointed", "disappointing", "frustrated", "frustrating", "let down", "not as expected",
    "didn't meet expectations", "not satisfied", "poor experience", "not impressed",
    "poor", "worst", "bad", "terrible", "cheap", "low quality", "inferior", "defective",
    "fake", "unreliable", "not reliable", "not durable", "not sturdy", "broke", "cracked", "crack",
    "disconnect", "connection failed", "bluetooth issue", "connectivity issue", "pairing issue",
    "not connecting", "won't connect", "interrupted", "network issue", "no signal",
    "confusing", "hard to use", "inconvenient", "unintuitive", "buggy", "glitch", "unresponsive",
    "uncomfortable", "hurts", "painful", "not ergonomic", "not fitting", "too tight", "too loose",
    "not worth", "not worthy", "not value for money", "overpriced", "unworthy", "scam", "cheated",
    "never again", "won't buy again", "don't recommend", "regret buying", "had to return", "returning it",
    "wouldn't recommend", "not recommendable", "total failure", "utter failure"
]


# ======================= LOAD MODEL ============================
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("saved_model")
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
    return pipe, tokenizer


pipe, tokenizer = load_model()

# ======================= APP UI ============================
st.set_page_config(page_title="User Frustration Predictor", page_icon="😠", layout="centered")
st.title("🤖 User Frustration Prediction")
st.markdown("Enter a product review, and this app will tell you whether the customer was **frustrated** or not.")

# ======================= REVIEW INPUT ============================
review = st.text_area("📝 Write your review here:", height=150)

if st.button("Predict Frustration Level"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        prediction = pipe(review)[0]
        label = prediction['label']
        score = prediction['score']

        if label == "LABEL_1":
            st.error(f"😠 Frustrated (Confidence: {score:.2%})")
        else:
            st.success(f"😊 Not Frustrated (Confidence: {score:.2%})")

# ======================= FILE UPLOAD ============================
st.markdown("## 📄 Bulk Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type="csv")


# ======================= TRIGGER HIGHLIGHTER ============================
def highlight_triggers(text):
    try:
        text = str(text)
        for phrase in sorted(frustration_keywords, key=lambda x: -len(x)):
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            text = pattern.sub(
                r"<span style='background-color:#ffb3b3;padding:3px 8px;border-radius:10px;font-weight:600;color:#a10000;'>\g<0></span>",
                text
            )
        return text
    except:
        return text


# ======================= SAFE PREDICTOR ============================
def predict_label_safe(text):
    try:
        tokens = tokenizer.tokenize(str(text))
        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
            text = tokenizer.convert_tokens_to_string(tokens)
        result = pipe(text)[0]['label']
        return "Frustrated" if result == "LABEL_1" else "Not Frustrated"
    except Exception as e:
        return f"ERROR: {str(e)}"


# ======================= BULK HANDLER ============================
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.markdown("Select the column that contains reviews:")
        review_column = st.selectbox("Column with review text", df.columns.tolist())

        df['token_count'] = df[review_column].apply(lambda x: len(tokenizer.tokenize(str(x))))
        long_count = df[df['token_count'] > MAX_LEN].shape[0]
        if long_count > 0:
            st.warning(f"⚠️ {long_count} reviews were longer than {MAX_LEN} tokens and were truncated.")

        with st.spinner("🔍 Analyzing reviews..."):
            df['Frustration_Prediction'] = df[review_column].apply(predict_label_safe)
            df['Highlighted_Review'] = df[review_column].apply(highlight_triggers)

        # ======================= CHARTS ============================
        st.success("✅ Predictions complete!")
        st.markdown("### 📊 Frustration Distribution")
        label_counts = df['Frustration_Prediction'].value_counts().reset_index()
        label_counts.columns = ['Frustration_Prediction', 'Count']

        fig_pie = px.pie(label_counts, names='Frustration_Prediction', values='Count',
                         title='Frustration Breakdown', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### 📈 Frustration Frequency Bar Chart")
        fig_bar = px.bar(label_counts, x='Frustration_Prediction', y='Count',
                         color='Frustration_Prediction', color_discrete_sequence=px.colors.qualitative.Set2,
                         text='Count')
        st.plotly_chart(fig_bar, use_container_width=True)

        # ======================= HIGHLIGHTED REVIEWS ============================
        st.markdown("### 🔍 Highlighted Reviews with Predictions")
        for _, row in df.head(10).iterrows():
            st.markdown(f"<h4 style='margin-bottom:4px;'>Prediction:</h4>", unsafe_allow_html=True)

            color = "#e63946" if row["Frustration_Prediction"] == "Frustrated" else "#2a9d8f"
            label_html = f"""
                <div style='
                    display:inline-block;
                    padding:6px 12px;
                    background-color:{color};
                    color:white;
                    border-radius:20px;
                    font-weight:600;
                    margin-bottom:10px;'
                >
                    {row['Frustration_Prediction']}
                </div>
            """
            st.markdown(label_html, unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style='
                    background-color:#f0f0f0;
                    padding:15px;
                    border-radius:8px;
                    font-size:16px;
                    line-height:1.6;
                    font-family:Arial, sans-serif;
                    color:#222;
                '>{row['Highlighted_Review']}</div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<hr style='margin-top:25px;'>", unsafe_allow_html=True)


        # ======================= DOWNLOAD ============================
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')


        csv = convert_df(df)
        st.download_button("📥 Download Results as CSV", data=csv,
                           file_name="frustration_predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Something went wrong: {str(e)}")

# ======================= FOOTER ============================
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with ❤️ by Ashmita Sharma</div>", unsafe_allow_html=True)
