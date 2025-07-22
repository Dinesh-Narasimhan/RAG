import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import docx
import pdfplumber
import tempfile
import subprocess

# Use llama-cpp-python (TinyLlama GGUF)
from llama_cpp import Llama

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Download TinyLlama GGUF if not present
    gguf_path = "models/tinyllama-1b-chat.gguf"
    if not os.path.exists(gguf_path):
        os.makedirs("models", exist_ok=True)
        st.info("⏬ Downloading TinyLlama model...")
        url = "https://huggingface.co/codellama/tinyllama-1b-chat-hf/resolve/main/TinyLlama-1B-Chat.gguf"
        subprocess.run(["wget", "-O", gguf_path, url], check=True)

    llm = Llama(model_path=gguf_path, n_ctx=1024, n_threads=4)
    return embedder, llm

embedder, llm = load_models()

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

st.title("🧠 Ask Your Notes (Lite RAG)")

uploaded_file = st.file_uploader("📂 Upload .txt, .docx, or .pdf", type=["txt", "docx", "pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    full_text = read_file(file_path, ext)

    chunks = chunk_text(full_text)
    st.success(f"✅ Document split into {len(chunks)} chunks")

    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    question = st.text_input("❓ Ask your question:")
    if question:
        question_embedding = embedder.encode([question])
        D, I = index.search(np.array(question_embedding), k=1)
        retrieved_chunk = chunks[I[0][0]]

        prompt = f"Context:\n{retrieved_chunk}\n\nQ: {question}\nA:"

        with st.spinner("🧠 Generating answer..."):
            result = llm(prompt, max_tokens=300, stop=["\n\n", "</s>"])
            st.markdown("### ✅ Answer:")
            st.write(result["choices"][0]["text"].strip())
