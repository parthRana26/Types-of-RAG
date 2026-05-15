# Multi-Query + Parent RAG + Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()


# Load PDF
loader = PyPDFLoader("docs/DL research paper.pdf")
docs = loader.load()


# Splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# FAISS setup
index = faiss.IndexFlatL2(384)

vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

store = InMemoryStore()


# Parent Retriever
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

parent_retriever.add_documents(docs)


# LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)


# Multi-Query Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=parent_retriever,
    llm=llm
)


# Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say "Not found in document".

Context:
{context}

Question:
{question}

Answer:
"""
)


# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=multi_query_retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)


# Query Loop
print("\n🔥 Multi-Query RAG Ready! Type 'exit' to quit\n")

while True:
    query = input("💬 Ask: ")

    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})

    print("\n🧠 Answer:\n")
    print(result["result"])

    print("\n📄 Sources:\n")
    for i, doc in enumerate(result["source_documents"]):
        print(f"[{i+1}] {doc.page_content[:200]}\n")

    print("=" * 50)