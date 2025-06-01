# Cloud-Friendly Text-to-Summary App (No Whisper/Audio)

import streamlit as st
from transformers import pipeline
from fpdf import FPDF
from datetime import datetime, timedelta
import os
import unicodedata

# === Setup folders ===
os.makedirs("outputs/pdf", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

# === Cleanup old PDFs and logs ===
def cleanup_old_files(folder, days=2):
    cutoff = datetime.now() - timedelta(days=days)
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(path))
            if mod_time < cutoff:
                os.remove(path)

cleanup_old_files("outputs/pdf")
cleanup_old_files("outputs/logs")

# === Load summarizer model ===
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# === PDF Export Function ===
def export_summary_to_pdf(summary_text, transcript_text):
    summary_text = unicodedata.normalize("NFKD", summary_text).encode("latin-1", "ignore").decode("latin-1")
    transcript_text = unicodedata.normalize("NFKD", transcript_text).encode("latin-1", "ignore").decode("latin-1")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pdf_path = os.path.join("outputs/pdf", f"summary_{timestamp}.pdf")

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Text Summary Report", ln=True, align="C")
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Summary:", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.ln(5)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Full Text:", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, transcript_text)

    pdf.output(pdf_path)
    return pdf_path

# === Log Function ===
def log_entry(filename, summary):
    log_path = os.path.join("outputs/logs", "activity_log.txt")
    with open(log_path, "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Generated: {filename}\n")
        log.write(f"Summary: {summary[:150]}...\n\n")

# === View logs ===
def display_logs():
    log_path = os.path.join("outputs/logs", "activity_log.txt")
    if os.path.exists(log_path):
        with open(log_path, "r") as log:
            logs = log.read()
        st.text_area("📜 Log History", logs, height=200)
    else:
        st.info("No logs yet.")

# === UI Starts Here ===
st.title("📝 Text to Summary Generator")
st.markdown("Paste your text or upload a `.txt` file. This app summarizes it using a Hugging Face model and lets you export as PDF.")

text_input = st.text_area("📋 Paste Your Text Here", height=250)

uploaded_file = st.file_uploader("📄 Or Upload a .txt File", type=["txt"])
if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")

if text_input:
    st.info("Summarizing...")
    summary = summarizer(text_input[:1000], max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    st.success("Summary generated!")
    st.text_area("🧾 Summary", summary, height=150)

    if st.button("📥 Export to PDF"):
        pdf_path = export_summary_to_pdf(summary, text_input)
        log_entry(os.path.basename(pdf_path), summary)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")

st.markdown("---")
st.subheader("📂 Export History & Logs")
display_logs()
