import os
from neo4j import GraphDatabase

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

docs = [
    Document(page_content="Elon Musk founded SpaceX."),
    Document(page_content="Elon Musk is CEO of Tesla."),
    Document(page_content="Tesla builds electric vehicles."),
    Document(page_content="SpaceX works on rockets and space missions.")
]

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = FAISS.from_documents(docs, embedding)

corpus = [doc.page_content.split() for doc in docs]
bm25 = BM25Okapi(corpus)

# Hybrid Search Function
def hybrid_search(query, k=3):
    # Vector search
    vector_results = vector_db.similarity_search(query, k=k)

    # BM25 search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_results = sorted(
        zip(docs, bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    bm25_docs = [doc for doc, _ in bm25_results]

    # Merge results
    combined = list({doc.page_content: doc for doc in vector_results + bm25_docs}.values())
    
    return combined

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Extract Triples
def extract_triples(text):
    prompt = f"""
    Extract knowledge triples from the text.
    Format: (subject, relation, object)

    Text:
    {text}
    """

    response = llm.invoke(prompt).content
    
    triples = []
    for line in response.split("\n"):
        if "(" in line and ")" in line:
            triple = line.strip("()").split(",")
            if len(triple) == 3:
                triples.append(tuple([t.strip() for t in triple]))
    
    return triples

# Insert into Neo4j
def insert_triples(triples):
    with driver.session() as session:
        for subj, rel, obj in triples:
            query = f"""
            MERGE (a:Entity {{name: $subj}})
            MERGE (b:Entity {{name: $obj}})
            MERGE (a)-[:{rel.replace(" ", "_").upper()}]->(b)
            """
            session.run(query, subj=subj, obj=obj)

# Ingest Documents
docs = [
    "Elon Musk founded SpaceX.",
    "Elon Musk is CEO of Tesla.",
    "Tesla builds electric vehicles.",
    "SpaceX develops rockets."
]

for doc in docs:
    triples = extract_triples(doc)
    insert_triples(triples)

# Query Graph (Cypher)
def query_graph(entity):
    with driver.session() as session:
        query = """
        MATCH (a:Entity {name: $entity})-[r]->(b)
        RETURN a.name, type(r), b.name
        """
        result = session.run(query, entity=entity)
        
        return [f"{r['a.name']} - {r['type(r)']} -> {r['b.name']}" for r in result]

# Graph RAG Pipeline
def graph_rag(query):
    # Extract entity from query (simple version)
    entity = "Elon Musk"  # replace with LLM extraction later

    # Get graph data
    relations = query_graph(entity)

    context = "\n".join(relations)

    prompt = f"""
    Answer the question using the knowledge graph.

    Graph Data:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    return response.content

query = "What companies are related to Elon Musk?"

print(graph_rag(query))