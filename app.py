# ✅ Streamlit Cloud-Compatible RAG App (using GPT-2)
# ⚡ Lightweight, Open-Source & Free

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import faiss
import os
import docx
import pdfplumber  # ✅ Lighter PDF reader
import tempfile

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # ✅ ~22MB
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # ✅ MIT licensed
    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cpu")
    model.eval()
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

def read_file(file_path, ext):
    if ext == ".txt":
        return open(file_path, "r", encoding="utf-8").read()
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() or '' for page in pdf.pages])
    return ""

def chunk_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

st.title("📄 RAG Question Answering (GPT-2 + MiniLM-L3)")
uploaded_file = st.file_uploader("Upload a .txt, .docx, or .pdf file", type=["txt", "docx", "pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    full_text = read_file(file_path, ext)

    chunks = chunk_text(full_text)
    st.success(f"✅ Text split into {len(chunks)} chunks")

    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    question = st.text_input("Ask a question:")
    if question:
        question_embedding = embedder.encode([question])
        D, I = index.search(np.array(question_embedding), k=1)
        retrieved_chunk = chunks[I[0][0]]

        prompt = f"""You are a helpful assistant. Using the context below, answer the question in detail.

Context: {retrieved_chunk}

Question: {question}
Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded[len(prompt):].strip()

        st.markdown("### 🧠 Answer")
        st.write(answer)
