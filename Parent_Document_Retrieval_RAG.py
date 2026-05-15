from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

# 1. Load PDF
loader = PyPDFLoader("docs/DL research paper.pdf")
docs = loader.load()

# 2. Splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# 3. Embeddings + stores
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.IndexFlatL2(384)

vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
                    )
store = InMemoryStore()

# 4. Parent-child retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 5. Add documents
retriever.add_documents(docs)

# 6. LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# 7. QA Chain
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant. Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# 8. Query Loop
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
    