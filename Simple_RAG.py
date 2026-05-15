from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

# Load PDF
loader = PyPDFLoader("docs/DL research paper.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
db = Chroma.from_documents(chunks, embedding)

retriever = db.as_retriever()

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context:

{context}

Question: {question}
""")

# Chain (LCEL)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Run
query = "What is ANN?"
response = chain.invoke(query)

print(response.content)