from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Load file
loader = PyPDFLoader("docs/DL research paper.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents, embedding)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Query Decomposition
def decompose_query(query):
    prompt = f"""
        Break this question into smaller sub-questions for reasoning.

        Question:
        {query}

        Sub-questions:
    """
    response = llm.invoke(prompt)
    
    sub_queries = response.content.split("\n")
    return [q.strip("- ").strip() for q in sub_queries if q.strip()]

# Multi-Query Retrieval
def retrieve_docs(sub_queries):
    all_docs = []
    
    for q in sub_queries:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)
    
    return all_docs

# Simple Re-ranking
def deduplicate_docs(docs):
    seen = set()
    unique_docs = []
    
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    
    return unique_docs

# Reasoning Prompt

reasoning_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert AI assistant.

Follow these steps:
1. Understand the problem
2. Analyze all relevant information
3. Think step-by-step
4. Compare if needed
5. Give final clear answer

Context:
{context}

Question:
{question}

Answer (with reasoning):
"""
)

# Final Reasoning RAG Pipeline
def reasoning_rag_pipeline(query):
    
    # Step 1: Decompose
    sub_queries = decompose_query(query)
    print("Sub-Queries:", sub_queries)
    
    # Step 2: Retrieve
    docs = retrieve_docs(sub_queries)
    
    # Step 3: Deduplicate
    docs = deduplicate_docs(docs)
    
    # Step 4: Build context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Step 5: Reasoning
    final_prompt = reasoning_prompt.format(
        context=context,
        question=query
    )
    
    response = llm.invoke(final_prompt)
    
    return response.content


query = "Compare ANN and RNN"

result = reasoning_rag_pipeline(query)

print(result)