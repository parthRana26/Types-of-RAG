from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Connect Graph
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

CYPHER_GENERATION_PROMPT = PromptTemplate.from_template("""
You are a Neo4j expert.

Generate Cypher query based on question.

IMPORTANT:
- Always return BOTH entity and relationship info
- Include person and project together

Schema:
{schema}

Question:
{question}

Cypher Query:
""")

CYPHER_QA_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant.

Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{question}

Give a direct and clear answer:
""")

# Create Chain
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    qa_prompt=CYPHER_QA_PROMPT,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    return_direct=False
)

query = "What project does Parth work on?"
response = chain.invoke({"query": query})

print(response["result"])