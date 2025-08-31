import streamlit as st
import fitz 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_ID = "google/bigbird-pegasus-large-arxiv"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)


if "document" not in st.session_state:
    st.session_state.document = ""

st.title("AI Summarizer")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if st.button("Upload File"):
    if uploaded_file is None:
        st.error("Please upload a file first")
    else:
        try:
            whole_doc = ""   
            with fitz.open(uploaded_file) as doc:
                whole_doc = "\n\n".join(
                    f"=== Page {i+1} ===\n{page.get_text('text') or ''}"
                    for i, page in enumerate(doc)
                )
            st.session_state.document = whole_doc
            st.success("File uploaded")
            
        except Exception as e:
            st.error("Could not process file")


if len(st.session_state.document) > 0:
    st.write(st.session_state.document)

# text = []
# with fitz.open("file.pdf") as doc:
#     for page in doc:
#         text.append(page.get_text()) 
# plain = "\n".join(text)