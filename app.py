import streamlit as st
import fitz  # PyMuPDF
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from fpdf import FPDF
import os

# --- Caching for the Model ---
# Caching the model loading prevents it from being reloaded every time the user interacts with the app.
@st.cache_resource
def load_model():
    """Loads and caches the translation model and tokenizer."""
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    return model, tokenizer

# --- Core Functions (from your original script) ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def translate_french_to_english(text, model, tokenizer):
    """Translates French text to English."""
    tokenizer.src_lang = "fr_XX"
    encoded_french = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    generated_tokens = model.generate(
        **encoded_french,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text[0]

def create_pdf_with_text(text):
    """Creates a PDF file in memory from the translated text."""
    pdf = FPDF()
    pdf.add_page()
    # Add a font that supports a wide range of characters (DejaVu is a good choice)
    # You'll need to provide the font file for this to work in a deployed environment.
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
    except RuntimeError:
        # Fallback to a standard font if DejaVu is not found
        st.warning("DejaVu font not found. Falling back to Arial. Special characters may not render correctly.")
        pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin-1')

# --- Streamlit App Interface ---
st.title("🇫🇷 French to 🇬🇧 English PDF Translator")
st.write("Upload a PDF file in French, and this app will translate it into English for you using a Large Language Model.")

# Load the model and tokenizer
model, tokenizer = load_model()

# File uploader widget
uploaded_file = st.file_uploader("Choose a French PDF file", type="pdf")

if uploaded_file is not None:
    # Button to start the translation
    if st.button("Translate PDF"):
        with st.spinner("Translating... This may take a moment."):
            # Step 1: Extract text from the uploaded PDF
            french_text = extract_text_from_pdf(uploaded_file)
            st.write("### Original French Text (First 500 characters)")
            st.text_area("", french_text[:500], height=150)

            # Step 2: Translate the text
            english_text = translate_french_to_english(french_text, model, tokenizer)
            st.write("### Translated English Text")
            st.text_area("", english_text, height=250)

            # Step 3: Create the translated PDF in memory
            pdf_output = create_pdf_with_text(english_text)

            # Step 4: Provide a download button for the translated PDF
            st.download_button(
                label="Download Translated PDF",
                data=pdf_output,
                file_name=f"translated_{uploaded_file.name}",
                mime="application/pdf"
            )

st.info("Note: The translation quality depends on the underlying model. Large and complex PDFs may take longer to process.")```