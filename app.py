import streamlit as st
import pandas as pd
from transformers import pipeline
from io import BytesIO
import plotly.express as px

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
        df = pd.read_csv(uploaded_file, encoding='utf-8')
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

        # ✅ عرض النتائج
        st.success("✅ Analysis complete!")
        st.dataframe(df, use_container_width=True)

        # ✅ رسم شريطي لتوزيع المشاعر
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                     title="Number of Each Sentiment", text="Count")
        st.plotly_chart(fig, use_container_width=True)

        # ✅ جدول تكرار الجمل
        st.subheader("🗂️ Text Frequency Table")
        text_counts = df["text"].value_counts().reset_index()
        text_counts.columns = ["text", "Count"]
        text_counts["Percentage"] = round((text_counts["Count"] / len(df)) * 100, 2)
        st.dataframe(text_counts)

        # ✅ تحميل Excel فقط
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Sentiment Results")
            writer.sheets["Sentiment Results"].right_to_left()
        excel_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Results as Excel",
            data=excel_buffer.getvalue(),
            file_name="sentiment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
