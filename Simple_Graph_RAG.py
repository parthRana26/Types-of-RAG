from neo4j import GraphDatabase
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

print("Connecting to Neo4j...")
# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

print("Connected to Neo4j")

# Create Graph Data
def create_graph(tx):
    tx.run("""
    MERGE (p:Person {name: 'Parth'})
    MERGE (proj:Project {name: 'Graph RAG'})
    MERGE (m:Manager {name: 'Alice'})
    
    MERGE (p)-[:WORKS_ON]->(proj)
    MERGE (p)-[:MANAGED_BY]->(m)
    """)

with driver.session() as session:
    session.execute_write(create_graph)


# Query the Graph (Traversal)
def get_manager(tx):
    result = tx.run("""
    MATCH (p:Person {name: 'Parth'})-[:MANAGED_BY]->(m:Manager)
    RETURN m.name AS manager
    """)
    
    return [record["manager"] for record in result]

with driver.session() as session:
    manager = session.execute_read(get_manager)

print(manager)


# Add LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)


# Graph RAG function
def graph_rag(query):
    
    # Step 1: Retrieve from Graph
    with driver.session() as session:
        result = session.run("""
        MATCH (p:Person)-[:WORKS_ON]->(proj:Project)
        MATCH (p)-[:MANAGED_BY]->(m:Manager)
        RETURN p.name AS person, proj.name AS project, m.name AS manager
        """)
        
        context = ""
        for r in result:
            context += f"{r['person']} works on {r['project']} and is managed by {r['manager']}.\n"
    
    # Step 2: Send to LLM
    prompt = f"""
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    
    response = llm.invoke(prompt)
    
    return response.content

# Test
query = "Parth project name ?"
print(graph_rag(query))