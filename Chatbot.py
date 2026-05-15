# Ultimate RAG: Hybrid + Parent + MultiQuery + Reranker

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from sentence_transformers import CrossEncoder
import faiss
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


# BM25 (Sparse)
bm25 = BM25Retriever.from_documents(docs)
bm25.k = 5


# Dense Retriever
dense = vectorstore.as_retriever(search_kwargs={"k": 5})


# Hybrid Retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.5, 0.5]
)


# Multi-Query Retriever
multi_query = MultiQueryRetriever.from_llm(
    retriever=hybrid_retriever,
    llm=ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
)


# Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_docs(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)


# Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer ONLY using the context below.
If not found, say "Not found".

Context:
{context}

Question:
{question}

Answer:
"""
)


# Query Loop
print("\n🔥 Ultimate RAG Ready!\n")

while True:
    query = input("💬 Ask: ")

    if query.lower() == "exit":
        break

    # Step 1: MultiQuery + Hybrid Retrieval
    docs = multi_query.get_relevant_documents(query)

    # Step 2: Rerank
    docs = rerank_docs(query, docs)

    # Step 3: Build Context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 4: LLM
    response = llm.invoke(prompt.format(context=context, question=query))

    print("\n🧠 Answer:\n")
    print(response.content)

    print("\n📄 Sources:\n")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content[:20]}\n")

    print("=" * 50)