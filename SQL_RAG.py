from urllib import response

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)


db = SQLDatabase.from_uri("sqlite:///company.db")
print(db.get_table_info())



chain = create_sql_query_chain(llm, db)

query = "Who has the highest salary?"

sql_query = chain.invoke({"question": query})

print("Generated SQL:", sql_query)


# SQL RAG Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

response = agent.run("List top 2 highest paid employees")
print(response)