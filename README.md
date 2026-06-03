# 🚀 Types of RAG (Retrieval-Augmented Generation)

This repository showcases implementations of advanced Retrieval-Augmented Generation (RAG) techniques, built using LangChain, HuggingFace embeddings, and LLM APIs. It demonstrates how different retrieval strategies improve response accuracy and reasoning in AI systems.

---

## 📌 Overview

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by combining retrieval mechanisms with generation. This project explores multiple RAG architectures with practical implementations.

---

## 🧠 RAG Techniques Covered

- ✅ Simple RAG  
- ✅ Hybrid RAG (Dense + BM25)  
- ✅ Parent-Child Retrieval RAG  
- ✅ Multi-Query RAG  
- ✅ Graph RAG  
- ✅ Hybrid + Graph RAG  
- ✅ SQL RAG  
- ✅ Reasoning-Based RAG (CoT-based)
- - ✅ Self RAG

---

## 📂 Project Structure

```
RAG/
│
├── Chatbot.py
├── Simple_RAG.py
├── Hybrid_RAG.py
├── Graph_RAG.py
├── Simple_Graph_RAG.py
├── Combine_Graph_Hybrid_RAG.py
├── SQL_RAG.py
├── Multi_Query_RAG.py
├── Parent_Document_Retrieval_RAG.py
├── Reasoning_Based_RAG.py
├── self_RAG.ipynb
│
├── create_db.py
├── company.db
├── sentence_transformers_1.py
│
├── docs/
├── .env
├── requirements.txt
└── rag_venv/
```

---

## ⚙️ Tech Stack

- 🦜 LangChain  
- 🤗 HuggingFace Embeddings  
- ⚡ Groq / LLM APIs  
- 🗄️ SQLite (for SQL RAG)  
- 🧠 Vector Databases (FAISS / others)  
- 🔗 Neo4j (for Graph RAG)
- 🔗 DDGS (DuckDuckGoSearch) Web Search

---

## 🔄 RAG Workflow

1. Load Documents (PDF, Text, DB, Graph)  
2. Split into Chunks  
3. Generate Embeddings  
4. Store in Vector DB / Graph / SQL  
5. Retrieve Relevant Context  
6. Pass Context to LLM  
7. Generate Final Answer  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Types-of-RAG.git
cd Types-of-RAG
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv rag_venv
source rag_venv/bin/activate   # Windows: rag_venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables
Create a `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run Examples

```bash
python Simple_RAG.py
python Hybrid_RAG.py
python Graph_RAG.py
python SQL_RAG.py
python Reasoning_Based_RAG.py
```

---

## 🔥 Key Learnings

- Difference between Fine-Tuning and RAG  
- Importance of retrieval quality  
- Combining multiple retrieval strategies  
- Using structured data (SQL, Graph) with LLMs  
- Enhancing reasoning with Chain-of-Thought  

---

## 📊 Future Improvements

- Add evaluation metrics (RAGAS, etc.)  
- Build UI (Streamlit / React)  
- Add API endpoints (FastAPI)  
- Optimize retrieval latency  

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## ⭐ Support

If you found this helpful, give it a ⭐ on GitHub!
