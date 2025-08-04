***📘 RAG Tutor — Ask Your Notes — AI-Powered Question Answering from Documents (Powered by Phi-2)***

RAG Tutor is an intelligent, offline-capable application that enables users to ask rich, detailed questions directly from uploaded documents (TXT, PDF, DOCX). It uses Retrieval-Augmented Generation (RAG), combining the semantic retrieval power of MiniLM and the expressive capabilities of Phi-2, Microsoft’s open-source large language model.

---

🚀 Key Features
Context-Aware Q&A
Combines document retrieval and AI generation to deliver grounded, fact-based answers from your own files.

Long-Form Answer Generation
Powered by Phi-2 to produce detailed, 300+ word responses with clarity and depth.

Semantic Search with MiniLM
Uses sentence-level embeddings and FAISS indexing to understand and retrieve the most relevant text passages.

Multi-Format Document Support
Accepts .txt, .pdf, and .docx formats to ensure flexibility for various user needs.

No Internet Required (Post Setup)
Once the models are downloaded, all processing happens locally for enhanced privacy and speed.

---

💼 Ideal Use Cases
📚 Studying from lecture notes and academic materials

🧠 Exploring knowledge from large text sources

🏢 Internal business document intelligence

📖 Enhancing reading comprehension for educational use

---

🧠 How It Works
Document Ingestion
User uploads a document. Text is extracted and split into manageable chunks.

Semantic Embedding
Each chunk is embedded using a MiniLM transformer model for semantic meaning.

Vector Indexing
FAISS is used to build a fast, searchable index of the embeddings.

Question Answering
A user question is embedded and compared to the indexed chunks. The most relevant one is passed, along with the question, to Phi-2 to generate a thorough, human-like response.

---

🔍 Core Technologies
Transformers (Hugging Face) – for the Phi-2 language model

SentenceTransformers – for MiniLM embeddings

FAISS – for efficient semantic search

Streamlit – for a clean, interactive user interface

PyTorch – powering model inference

---

📌 Notes
Initial setup requires downloading the Phi-2 model (~1.7GB). After that, all operations can run locally.

Best performance is achieved with machines that have 8–16 GB RAM or more.

While GPU acceleration improves speed, the app is also compatible with CPU-only environments.

---

Acknowledgments
Microsoft for Phi-2

Hugging Face for Transformers and SentenceTransformers

Meta AI for FAISS

Streamlit for the frontend interface

---

Output:

<img width="1872" height="850" alt="Screenshot 2025-08-04 193033" src="https://github.com/user-attachments/assets/467497c2-ca65-41bc-bd2f-02122ef76e0d" />
