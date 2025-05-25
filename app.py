import streamlit as st
import pandas as pd
from transformers import pipeline

# إعداد الصفحة
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Sentiment Analysis")

# تحميل النموذج
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

model = load_model()

# رفع الملف
uploaded_file = st.file_uploader("📄 Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # قراءة الملف
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # تحقق من وجود العمود
    if "text" not in df.columns:
        st.error("❌ The file must contain a column named 'text'.")
    else:
        # تحليل المشاعر
        with st.spinner("Analyzing..."):
            predictions = model(df["text"].astype(str).tolist())
            df["Sentiment"] = [pred["label"] for pred in predictions]
            df["Confidence"] = [round(pred["score"], 2) for pred in predictions]

        # عرض النتائج
        st.success("✅ Analysis complete!")
        st.dataframe(df)

        # تحميل النتائج
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df)
        st.download_button(" Download Results as CSV", csv, "sentiment_results.csv", "text/csv")
