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

@torch.inference_mode()
def summarize_long(text: str, max_new_tokens=256):
    chunks = chunk_by_tokens(text, tokenizer)
    partials = [] #stores per-chunk summaries

    #Model settings
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=4,
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

st.title("AI Summarizer")

#upload & save file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if st.button("Upload File"):
    if uploaded_file is None:
        st.error("Please upload a file first")
    else:
        try:
            whole_doc = ""   
            file_bytes = uploaded_file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                whole_doc = "\n\n".join(
                    f"=== Page {i+1} ===\n{page.get_text('text') or ''}"
                    for i, page in enumerate(doc)
                )#seperate pages by ===Page #==== and new lines
            st.session_state.document = whole_doc #save document to the session state
            st.success("File uploaded")
            
        except Exception as e:
            st.error(e)

if len(st.session_state.document) > 0:
    st.write(st.session_state.document)
    st.subheader("Final summary:")
    final, partials = summarize_long(st.session_state.document, 256)
    st.write(final)
    with st.expander("Chunk summaries"):
        for i, p in enumerate(partials, 1):
            st.markdown(f"**Chunk {i}**")
            st.write(p)




