import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import faiss
import os
import docx
import pdfplumber
import tempfile

# Load embedding model & GPT-2 once
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")  # ~22MB
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cpu")
    model.eval()
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

# Read supported file types
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

# Chunk text (~100 words each)
def chunk_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Streamlit UI
st.title("📘 Ask Your Notes (Lightweight RAG using GPT-2)")

uploaded_file = st.file_uploader("📂 Upload your notes (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    full_text = read_file(file_path, ext)

    chunks = chunk_text(full_text)
    st.success(f"✅ File processed and split into {len(chunks)} chunks.")

    # Embed and index chunks
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    question = st.text_input("❓ Ask a question from your notes:")
    if question:
        q_embed = embedder.encode([question])
        _, I = index.search(np.array(q_embed), k=1)
        context = chunks[I[0][0]]

        # Better prompt to avoid repeating
        prompt = f"Based on the following context, explain in detail:\n\n{context}\n\nWrite a clear, helpful explanation (at least 300 words) without repeating the same phrases or the original question."

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                min_length=300,
                do_sample=True,
                top_k=30,
                top_p=0.85,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded[len(prompt):].strip()

        st.markdown("### 🧠 Answer")
        st.write(answer)
