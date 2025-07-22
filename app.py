import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import os
import docx
import pdfplumber
import tempfile

# Load models once
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # ✅ Lightweight
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1", trust_remote_code=True, cache_dir="models")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-1",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        cache_dir="models"
    ).to("cpu")
    model.eval()
    return embedder, tokenizer, model

embedder, tokenizer, model = load_models()

# Read file content
def read_file(file_path, ext):
    if ext == ".txt":
        return open(file_path, "r", encoding="utf-8").read()
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() or '' for page in pdf.pages])
    else:
        return ""

# Split into ~100-word chunks
def chunk_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# UI
st.title("🧠 Ask Your Notes with Phi-1 (Fast & Free RAG)")

uploaded_file = st.file_uploader("📂 Upload a .txt, .docx, or .pdf file", type=["txt", "docx", "pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    full_text = read_file(file_path, ext)

    if not full_text.strip():
        st.error("❌ No readable content found in the uploaded file.")
        st.stop()

    chunks = chunk_text(full_text)
    st.success(f"✅ Split into {len(chunks)} chunks.")

    embeddings = embedder.encode(chunks)
    question = st.text_input("❓ Ask your question:")

    if question:
        question_embedding = embedder.encode([question])
        similarities = cosine_similarity([question_embedding[0]], embeddings)[0]
        top_idx = int(np.argmax(similarities))
        retrieved_chunk = chunks[top_idx]

        prompt = f"""You are a knowledgeable tutor. Using the context provided, write a very detailed and comprehensive answer to the question below. Make sure the answer is clear, complete, and at least 300 words long.

Context: {retrieved_chunk}

Question: {question}
Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=700,
                min_length=450,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded[len(prompt):].strip()

        st.markdown("### 🧠 Answer")
        st.write(answer)
