# 1. Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv()


# 2. Load PDF
loader = PyPDFLoader("docs/DL research paper.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")


# 3. Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = splitter.split_documents(docs)

print(f"Created {len(documents)} chunks")


# 4. Dense Retriever (FAISS)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(documents, embedding)

dense_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# 5. Sparse Retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 4


# 6. Hybrid Retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]   # 🔥 Tune this
)


# 7. Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)


# 8. RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=hybrid_retriever,
    return_source_documents=True
)


# 9. Query Loop
print("\n🔥 Hybrid RAG Ready! Type 'exit' to quit\n")

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

    print("="*50)