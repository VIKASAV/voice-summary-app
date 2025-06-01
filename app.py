# Voice-to-Summary App Using Whisper + Hugging Face + Enhanced PDF Export with Organized File Storage, Cleanup, Cloud Sync, and Logs

import whisper
import streamlit as st
import os
from pydub import AudioSegment
from pydub.utils import which
AudioSegment.converter = which("ffmpeg")
from transformers import pipeline
from fpdf import FPDF
from datetime import datetime, timedelta
import unicodedata
import shutil

# === Create folders for outputs and logs ===
os.makedirs("outputs/audio", exist_ok=True)
os.makedirs("outputs/pdf", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

# === Cleanup files older than 2 days ===
def cleanup_old_files(folder, days=2):
    cutoff = datetime.now() - timedelta(days=days)
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(path))
            if mod_time < cutoff:
                os.remove(path)

cleanup_old_files("outputs/audio")
cleanup_old_files("outputs/pdf")

# === Load summarization model (Hugging Face) ===
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# === Transcribe Audio Function ===
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

# === Summarize Text Function ===
def summarize_text(text):
    if len(text) > 1000:
        text = text[:1000]  # Truncate if too long
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# === PDF Export Function ===
def export_summary_to_pdf(summary_text, transcript_text):
    summary_text = unicodedata.normalize("NFKD", summary_text).encode("latin-1", "ignore").decode("latin-1")
    transcript_text = unicodedata.normalize("NFKD", transcript_text).encode("latin-1", "ignore").decode("latin-1")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pdf_file_path = os.path.join("outputs/pdf", f"summary_{timestamp}.pdf")

    pdf = FPDF()
    pdf.add_page()

    # Title and timestamp
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Voice Summary Report", ln=True, align="C")
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    # Summary Section
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Summary:", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.ln(5)

    # Transcript Section
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Transcript:", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, transcript_text)

    pdf.output(pdf_file_path)
    return pdf_file_path

# === Logging Function ===
def log_entry(filename, summary):
    log_path = os.path.join("outputs/logs", "activity_log.txt")
    with open(log_path, "a") as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Generated: {filename}\n")
        log_file.write(f"Summary Preview: {summary[:150]}...\n\n")

# === Display Log Viewer ===
def display_logs():
    log_path = os.path.join("outputs/logs", "activity_log.txt")
    if os.path.exists(log_path):
        with open(log_path, "r") as log_file:
            logs = log_file.read()
            st.text_area("📜 Activity Log", logs, height=250)
    else:
        st.info("No logs available yet.")

# === Streamlit UI ===
st.title("🎙️ Voice-to-Summary Generator (Cloud-Ready Edition)")
st.markdown("Upload a voice recording (.mp3, .wav, or .m4a) and get a summarized report using Whisper + Hugging Face.")

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    audio_path = os.path.join("outputs/audio", f"input_{timestamp}.{uploaded_file.name.split('.')[-1]}")

    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Transcribing audio...")
    try:
        # Convert to .wav if needed
        if uploaded_file.name.endswith(".mp3"):
            audio = AudioSegment.from_mp3(audio_path)
            audio.export("converted.wav", format="wav")
            file_to_transcribe = "converted.wav"
        elif uploaded_file.name.endswith(".m4a"):
            audio = AudioSegment.from_file(audio_path, format="m4a")
            audio.export("converted.wav", format="wav")
            file_to_transcribe = "converted.wav"
        else:
            file_to_transcribe = audio_path

        transcript = transcribe_audio(file_to_transcribe)
        st.success("Transcription complete!")
        st.text_area("📝 Transcribed Text", transcript, height=200)

        st.info("Generating summary...")
        summary = summarize_text(transcript)
        st.success("Summary generated!")
        st.text_area("📋 Summarized Report", summary, height=300)

        # PDF Download Button
        if st.button("📥 Export Summary as PDF"):
            pdf_path = export_summary_to_pdf(summary, transcript)
            log_entry(os.path.basename(pdf_path), summary)
            with open(pdf_path, "rb") as f:
                st.download_button(label="Download PDF", data=f, file_name=os.path.basename(pdf_path), mime="application/pdf")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# === Show Activity Logs ===
st.markdown("---")
st.subheader("📂 Export History & Logs")
display_logs()
