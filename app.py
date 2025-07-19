import streamlit as st
import fitz  # PyMuPDF
from transformers import MarianMTModel, MarianTokenizer
from fpdf import FPDF
import os

# --- Caching for the Model ---
# Caching the model loading prevents it from being reloaded every time the user interacts with the app.
@st.cache_resource
def load_model():
    """Loads and caches the translation model and tokenizer."""
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

# --- Core Functions ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def translate_french_to_english(text, model, tokenizer):
    """Translates French text to English using the Helsinki-NLP model."""
    # Split the text into manageable chunks to avoid overwhelming the model
    chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
    translated_chunks = []
    
    for chunk in chunks:
        # Translate each chunk
        translated = model.generate(**tokenizer(chunk, return_tensors="pt", padding=True))
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_chunks.append(translated_text)
        
    return " ".join(translated_chunks)

def create_pdf_with_text(text):
    """Creates a PDF file in memory from the translated text."""
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
    except RuntimeError:
        st.warning("DejaVu font not found. Falling back to Arial. Special characters may not render correctly.")
        pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin-1')

# --- Streamlit App Interface ---
st.title("🇫🇷 French to 🇬🇧 English PDF Translator")
st.write("Upload a PDF file in French, and this app will translate it into English using a specialized language model.")

# Load the model and tokenizer
with st.spinner("Loading translation model..."):
    model, tokenizer = load_model()

# File uploader widget
uploaded_file = st.file_uploader("Choose a French PDF file", type="pdf")

if uploaded_file is not None:
    # Button to start the translation
    if st.button("Translate PDF"):
        with st.spinner("Translating... This may take a moment."):
            # Step 1: Extract text
            french_text = extract_text_from_pdf(uploaded_file)
            st.write("### Original French Text (First 500 characters)")
            st.text_area("", french_text[:500], height=150)

            # Step 2: Translate the text
            english_text = translate_french_to_english(french_text, model, tokenizer)
            st.write("### Translated English Text")
            st.text_area("", english_text, height=250)

            # Step 3: Create the translated PDF
            pdf_output = create_pdf_with_text(english_text)

            # Step 4: Provide a download button
            st.download_button(
                label="Download Translated PDF",
                data=pdf_output,
                file_name=f"translated_{uploaded_file.name}",
                mime="application/pdf"
            )

st.info("This app uses the Helsinki-NLP model for translation.")
