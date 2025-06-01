import streamlit as st
import pandas as pd
from transformers import pipeline
from io import BytesIO

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

    # تحقق من وجود عمود text
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
        st.dataframe(df, use_container_width=True)

        # ⬇️ زر تحميل CSV
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_data,
            file_name="sentiment_results.csv",
            mime="text/csv"
        )

        # ⬇️ زر تحميل Excel
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Sentiment Results")
            worksheet = writer.sheets["Sentiment Results"]
            worksheet.right_to_left()  # دعم RTL
        excel_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Results as Excel",
            data=excel_buffer.getvalue(),
            file_name="sentiment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
