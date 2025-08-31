**How to run:**  
Open the terminal at the project root and type in the command:
```
streamlit run front.py
```
This should open your localhost with the program (it may take a couple seconds to load)  
Once The page is open, simply upload a pdf into the indicated slot (may take some time to process)  
Press summarize once the desired file is uploaded and wait for the AI to respond (will generally take a while especially for longer documents)

**Requirements (Python 3.12.10):**
- streamlit>=1.36
- pymupdf>=1.24
- transformers>=4.41
- sentencepiece>=0.1.99
- torch>=2.3

**Additional files:**
Some additional files have been provided however they are not the best quality for the project.  
files that are too short, have text in multiple columns, contain multiple different topics, or have many images may not provide the best results
