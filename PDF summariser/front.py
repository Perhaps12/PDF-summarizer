import streamlit as st
import fitz 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

#Ai model setup
#bigbird-pegasus can have inputs of up to 4096 tokens
MODEL_ID = "google/bigbird-pegasus-large-arxiv"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)

def pdf_to_text(file):
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            pages.append(page.get_text("text") or "")
    return "\n\n".join(pages)

#split text into chunks
def chunk_by_tokens(text: str, tokenizer, max_input_tokens=3800):
    # method intended to split the document into chunks of text each about 3800 tokens
    # leave headroom for special tokens; ~3800 is safe under 4096
    words = text.split() #split the document by spaces, newlines, tabs, etc.
    chunks, cur_words = [], []

    #returns the number of tokens in a list of words
    def tok_len(words_):
        return len(tokenizer.encode(" ".join(words_), add_special_tokens=False)) 

    for w in words:
        cur_words.append(w)
        #add words until the token surpasses the threshold
        if tok_len(cur_words) >= max_input_tokens:
            cur_words.pop()
            chunks.append(" ".join(cur_words)) #add current text to a new chunk
            cur_words = [w]# append the threshold breaking word to a new list
    if cur_words:
        chunks.append(" ".join(cur_words)) #append any remaining words if they exist
    return chunks

#summarization
@torch.inference_mode()
def summarize_long(text: str, max_new_tokens=256):
    chunks = chunk_by_tokens(text, tokenizer)
    partials = [] #stores per-chunk summaries

    #Model settings
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=2,
        no_repeat_ngram_size=3,
        length_penalty=0.8,
        early_stopping=True,
    )

    for c in chunks:
        inputs = tokenizer(c, return_tensors="pt", truncation=True).to(device) #tokenise each chunk and set to cuda if avaliable
        ids = model.generate(**inputs, **gen_kwargs)    #generate output
        partials.append(tokenizer.decode(ids[0], skip_special_tokens=True)) #decode and append partial summary to partials

    # second pass to condense the chunk summaries
    joined = "\n".join(partials) #join together artial summaries
    inputs = tokenizer(joined, return_tensors="pt", truncation=True).to(device)
    ids = model.generate(**inputs, **gen_kwargs) #tokenise and generate final summary
    return tokenizer.decode(ids[0], skip_special_tokens=True), partials #returns tuple of final summary and partial summaries

if "document" not in st.session_state:
    st.session_state.document = ""

st.title("RapidRead Summarizer")
#upload & save file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    if st.button("Summarize"):
        with st.spinner("Extracting text..."):
            file_bytes = uploaded_file.read()
            text = pdf_to_text(file_bytes)
        if not text.strip():
            st.warning("No extractable text found..")
        else:
            with st.spinner("Summarizing... (first run may take longer)"):
                final, partials = summarize_long(text, 512)
            st.subheader("Final Summary")
            st.write(final)
            with st.expander("Chunk summaries (summary every ~3000 words)"):
                for i, p in enumerate(partials, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(p)





